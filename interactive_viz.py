#!/usr/bin/env python
"""
Interactive visualization for the conditioned Flow Matching model.

Opens a Dash web app in your browser. Two sliders control the conditioning
observation (obs_x, obs_y). Every time you move a slider the model runs
live inference and updates all trajectory panels.

Usage:
    python interactive_viz.py
    python interactive_viz.py --ode-steps 10   # faster, rougher
    python interactive_viz.py --n-points 3000
    python interactive_viz.py --checkpoint-dir checkpoints_fm_cond

Then open http://127.0.0.1:8050 in your browser.
"""
from __future__ import print_function

import argparse
import os
os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "1")

import numpy as np
import tensorflow as tf
import tf_keras

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flow_matching import VelocityField, make_checkpoint_manager, settings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_ODE_STEPS = 20
DEFAULT_N_POINTS  = 2000
MAX_N_POINTS      = 5000   # pre-generated; slider subsets this
SNAP_T_VALUES     = [0.0, 0.25, 0.5, 0.75, 1.0]   # t values shown as panels
N_PANELS          = len(SNAP_T_VALUES)
OBS_RANGE         = settings['obs_range']
LIM               = settings['plot_axis_limit']

# Time schedule power: t_i = (i/n)^TIME_POWER
#   1.0 → uniform  |  0.7 → moderate bias toward t=1  |  0.5 → strong  |  →0 → halving
TIME_POWER = 0.7

QUAD_COLORS = {
    (True,  True):  '#e05555',   # red    (x<0, y<0)
    (False, True):  '#55bb55',   # green  (x>0, y<0)
    (True,  False): '#5588ee',   # blue   (x<0, y>0)
    (False, False): '#444444',   # dark   (x>0, y>0)
}
QUAD_NAMES = {
    (True,  True):  'x<0, y<0',
    (False, True):  'x>0, y<0',
    (True,  False): 'x<0, y>0',
    (False, False): 'x>0, y>0',
}

# ---------------------------------------------------------------------------
# Model + fixed noise (module-level so the Dash callback can reach them)
# ---------------------------------------------------------------------------
_model      = None
_x0         = None
_qmasks     = None
_ode_runner = None   # compiled tf.function set in main()


def load_model(checkpoint_dir):
    model = VelocityField()
    dummy_obs = tf.zeros((1, settings['obs_dim']))
    model(tf.zeros((1, 2)), tf.zeros((1, 1)), dummy_obs)

    optimizer = tf_keras.optimizers.Adam(jit_compile=False)
    ckpt, manager, global_step = make_checkpoint_manager(
        model, optimizer, checkpoint_dir=checkpoint_dir)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print("Loaded checkpoint: {}  (step {:,})".format(
            manager.latest_checkpoint, int(global_step.numpy())))
    else:
        print("WARNING: no checkpoint found in '{}'. "
              "Model is randomly initialised — start training first.".format(checkpoint_dir))

    return model


def make_ode_runner(model):
    """
    Returns a @tf.function that runs the full ODE entirely on the GPU.

    The entire loop compiles to a single graph call — no Python overhead
    between steps.  snap_t_values is a tensor input so the same compiled
    graph works for any set of snapshot targets without retracing.

    Design:
      - Iterates over segments defined by consecutive snap_t_values entries
      - Uses steps_per_segment Euler steps within each segment
      - Writes each segment endpoint into a TensorArray
      - Returns (n_snaps, n_points, 2) stacked tensor

    Retracing only occurs when steps_per_segment changes (mouseup slider)
    or when n_points changes; obs and snap_t_values are free to vary.
    """
    @tf.function(reduce_retracing=True)
    def _run(x0, obs_tf, snap_t_values, steps_per_seg_tensor):
        """
        snap_t_values:      (n_snaps,)  float32 — snapshot t targets
        steps_per_seg_tensor: (n_segs,) int32   — steps to use in each segment

        The outer loop over segments is unrolled at trace time (Python loop over
        a fixed-length list), so each segment's inner tf.range loop compiles to
        an independent tf.while_loop on the GPU.
        """
        n_snaps = snap_t_values.shape[0]  # static at trace time
        out = tf.TensorArray(dtype=tf.float32, size=n_snaps)
        out = out.write(0, x0)
        x = x0

        for seg in range(n_snaps - 1):   # Python loop → unrolled at trace time
            t_start  = snap_t_values[seg]
            t_end    = snap_t_values[seg + 1]
            n_steps  = steps_per_seg_tensor[seg]
            dt = (t_end - t_start) / tf.cast(n_steps, tf.float32)

            for step in tf.range(n_steps):   # TF loop → single tf.while_loop on GPU
                t_now = t_start + tf.cast(step, tf.float32) * dt
                t_col = tf.fill((tf.shape(x)[0], 1), t_now)
                x = x + model(x, t_col, obs_tf, training=False) * dt

            out = out.write(seg + 1, x)

        return out.stack()   # (n_snaps, n_points, 2)

    return _run


def run_trajectory(obs_val, ode_steps, time_power, n_points):
    """
    Run the ODE entirely on the GPU via a compiled tf.function.

    steps_per_segment is computed in Python from ode_steps and time_power:
      - time_power=1.0: equal steps per segment
      - time_power<1.0: more steps allocated to later (higher-t) segments,
        concentrating integration effort near t=1

    Returns:
        snaps     — list of N_PANELS numpy arrays (one per SNAP_T_VALUES)
        steps_vec — per-segment step counts (for display)
    """
    n_segs = len(SNAP_T_VALUES) - 1

    # Allocate steps per segment using time_power: weight ∝ (1 - t_start)^(1-power)
    # power=1 → uniform; power→0 → all steps in last segment
    weights = np.array(
        [(1.0 - SNAP_T_VALUES[i]) ** (1.0 - time_power) for i in range(n_segs)],
        dtype=np.float64)
    weights /= weights.sum()
    steps_vec = np.maximum(1, np.round(weights * ode_steps).astype(int))
    steps_vec[-1] += ode_steps - steps_vec.sum()   # absorb rounding remainder

    obs_arr      = np.tile(np.array([obs_val], dtype=np.float32), (n_points, 1))
    snap_t_tf    = tf.constant(SNAP_T_VALUES, dtype=tf.float32)

    # Run each segment with its own step count.
    # Each call is compiled; steps_vec entries change only on mouseup → rare retrace.
    x0_tf  = tf.constant(_x0[:n_points], dtype=tf.float32)
    obs_tf = tf.constant(obs_arr)
    result = _ode_runner(x0_tf, obs_tf, snap_t_tf,
                        tf.constant(steps_vec, dtype=tf.int32))
    # result: (n_snaps, n_points, 2)

    snaps = [result[i].numpy() for i in range(len(SNAP_T_VALUES))]
    return snaps, steps_vec.tolist()


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

CLASSES = {
    'brad':  {'one_hot': [1, 0], 'label': 'BRAD'},
    'katie': {'one_hot': [0, 1], 'label': 'KATIE'},
}


def build_figure(obs_x, obs_y, dataset_class, ode_steps, time_power, n_points):
    one_hot  = CLASSES[dataset_class]['one_hot']
    obs_full = [obs_x, obs_y] + one_hot   # [dx, dy, is_brad, is_katie]
    snaps, steps_vec = run_trajectory(obs_full, ode_steps, time_power, n_points)
    qmasks = {k: v[:n_points] for k, v in _qmasks.items()}

    # Panel subtitles: target t + how many steps were used to get there
    subtitles = ['t = {:.2f}  ({} steps)'.format(t, s)
                 for t, s in zip(SNAP_T_VALUES, [0] + steps_vec)]

    fig = make_subplots(
        rows=1, cols=N_PANELS,
        subplot_titles=subtitles,
        horizontal_spacing=0.04,
    )

    for col, snap in enumerate(snaps, start=1):
        first_col = (col == 1)
        for (bx, by), color in QUAD_COLORS.items():
            mask = qmasks[(bx, by)]
            fig.add_trace(
                go.Scattergl(
                    x=snap[mask, 0].tolist(),
                    y=snap[mask, 1].tolist(),
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.55),
                    name=QUAD_NAMES[(bx, by)],
                    legendgroup=QUAD_NAMES[(bx, by)],
                    showlegend=first_col,
                ),
                row=1, col=col,
            )
        fig.update_xaxes(
            range=[-LIM, LIM], row=1, col=col,
            showgrid=True, gridcolor='#e0e0e0', zeroline=True, zerolinecolor='#aaaaaa',
            showticklabels=True, tickfont=dict(size=10),
            fixedrange=True,
        )
        fig.update_yaxes(
            range=[-LIM, LIM], row=1, col=col,
            showgrid=True, gridcolor='#e0e0e0', zeroline=True, zerolinecolor='#aaaaaa',
            showticklabels=True, tickfont=dict(size=10),
            fixedrange=True,
            scaleanchor='x{}'.format(col if col > 1 else ''),
            scaleratio=1,
        )

    # Show per-segment step allocation: e.g. "3+4+5+8 = 20 steps"
    seg_labels = ['{:.2f}→{:.2f}: {}steps'.format(SNAP_T_VALUES[i], SNAP_T_VALUES[i+1], s)
                  for i, s in enumerate(steps_vec)]
    sched_str = '  |  '.join(seg_labels)

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8f8f8',
        title=dict(
            text=(
                '<b>{}</b> &nbsp;&nbsp; obs = ({:.2f}, {:.2f}) &nbsp;&nbsp; '
                'steps={} &nbsp; power={:.2f} &nbsp; pts={}<br>'
                '<span style="font-size:10px; color:#888888">'
                '{}</span>'
            ).format(CLASSES[dataset_class]['label'],
                     obs_x, obs_y, ode_steps, time_power, n_points, sched_str),
            font=dict(size=14, color='#333344'),
            x=0.5,
        ),
        height=450,
        margin=dict(l=10, r=10, t=110, b=10),
        legend=dict(
            x=1.01, y=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333'),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def make_app():
    app = dash.Dash(__name__, title='Flow Matching Viz')
    row_style = {'display': 'flex', 'gap': '40px',
                 'padding': '0 40px', 'marginTop': '10px'}
    label_style_blue = {'color': '#4455cc', 'fontWeight': 'bold', 'fontSize': '13px'}
    label_style_red  = {'color': '#cc5533', 'fontWeight': 'bold', 'fontSize': '13px'}
    label_style_grey = {'color': '#557755', 'fontWeight': 'bold', 'fontSize': '13px'}
    label_style_purp = {'color': '#885599', 'fontWeight': 'bold', 'fontSize': '13px'}

    app.layout = html.Div(
        style={'backgroundColor': '#ffffff', 'fontFamily': 'sans-serif',
               'padding': '20px', 'minHeight': '100vh'},
        children=[
            # Wrapper keeps the graph in place; spinner is absolutely-positioned
            # in the top-right corner and does NOT wrap the graph, so the
            # old figure stays fully visible until the new one arrives.
            html.Div(
                style={'position': 'relative'},
                children=[
                    dcc.Graph(id='trajectory-plot', config={'displayModeBar': False}),
                    # Spinner anchor: fixed-size box in the corner so dcc.Loading
                    # has space to render the circle inside it.
                    html.Div(
                        style={
                            'position': 'absolute',
                            'top': '10px',
                            'right': '10px',
                            'width': '44px',
                            'height': '44px',
                            'zIndex': 1000,
                        },
                        children=dcc.Loading(
                            id='spinner',
                            type='circle',
                            color='#4455cc',
                            children=html.Div(id='loading-placeholder',
                                              style={'width': '44px', 'height': '44px'}),
                        ),
                    ),
                ],
            ),

            # Row 0: dataset selector
            html.Div(
                style={'padding': '10px 40px 0 40px', 'display': 'flex',
                       'alignItems': 'center', 'gap': '16px'},
                children=[
                    html.Label('Dataset:', style={'fontWeight': 'bold',
                                                   'fontSize': '14px',
                                                   'color': '#333344'}),
                    dcc.RadioItems(
                        id='radio-class',
                        options=[
                            {'label': ' BRAD',  'value': 'brad'},
                            {'label': ' KATIE', 'value': 'katie'},
                        ],
                        value='brad',
                        inline=True,
                        inputStyle={'marginRight': '4px'},
                        labelStyle={'marginRight': '20px', 'fontSize': '14px',
                                    'cursor': 'pointer'},
                    ),
                ],
            ),

            # Row 1: observation sliders
            html.Div(style=row_style, children=[
                html.Div(style={'flex': 1}, children=[
                    html.Label('obs x', style=label_style_blue),
                    dcc.Slider(
                        id='slider-x',
                        min=-OBS_RANGE, max=OBS_RANGE, step=0.05, value=0.0,
                        marks={v: {'label': str(v), 'style': {'color': '#8899ff'}}
                               for v in [-1, -0.5, 0, 0.5, 1]},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='drag',
                    ),
                ]),
                html.Div(style={'flex': 1}, children=[
                    html.Label('obs y', style=label_style_red),
                    dcc.Slider(
                        id='slider-y',
                        min=-OBS_RANGE, max=OBS_RANGE, step=0.05, value=0.0,
                        marks={v: {'label': str(v), 'style': {'color': '#ff9977'}}
                               for v in [-1, -0.5, 0, 0.5, 1]},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='drag',
                    ),
                ]),
            ]),

            # Row 2: integration sliders
            html.Div(style=row_style, children=[
                html.Div(style={'flex': 1}, children=[
                    html.Label('ODE steps  (more = slower but more accurate)',
                               style=label_style_grey),
                    dcc.Slider(
                        id='slider-steps',
                        min=1, max=100, step=1, value=DEFAULT_ODE_STEPS,
                        marks={v: {'label': str(v), 'style': {'color': '#557755'}}
                               for v in [1, 5, 10, 20, 50, 100]},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='mouseup',
                    ),
                ]),
                html.Div(style={'flex': 1}, children=[
                    html.Label('time power  (1.0 = uniform  →  0.1 = biased toward t=1)',
                               style=label_style_purp),
                    dcc.Slider(
                        id='slider-power',
                        min=0.1, max=1.0, step=0.05, value=TIME_POWER,
                        marks={v: {'label': str(v), 'style': {'color': '#885599'}}
                               for v in [0.1, 0.3, 0.5, 0.7, 1.0]},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='mouseup',
                    ),
                ]),
                html.Div(style={'flex': 1}, children=[
                    html.Label('num points',
                               style={'color': '#996633', 'fontWeight': 'bold',
                                      'fontSize': '13px'}),
                    dcc.Slider(
                        id='slider-npoints',
                        min=200, max=MAX_N_POINTS, step=200, value=DEFAULT_N_POINTS,
                        marks={v: {'label': str(v), 'style': {'color': '#996633'}}
                               for v in [200, 1000, 2000, 3000, MAX_N_POINTS]},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='mouseup',
                    ),
                ]),
            ]),
        ],
    )

    @app.callback(
        Output('trajectory-plot',     'figure'),
        Output('loading-placeholder', 'children'),   # drives the spinner
        Input('radio-class',    'value'),
        Input('slider-x',       'value'),
        Input('slider-y',       'value'),
        Input('slider-steps',   'value'),
        Input('slider-power',   'value'),
        Input('slider-npoints', 'value'),
    )
    def update(dataset_class, obs_x, obs_y, ode_steps, time_power, n_points):
        fig = build_figure(
            obs_x    or 0.0,
            obs_y    or 0.0,
            dataset_class or 'brad',
            int(ode_steps  or DEFAULT_ODE_STEPS),
            float(time_power or TIME_POWER),
            int(n_points or DEFAULT_N_POINTS),
        )
        return fig, ''   # empty string keeps the placeholder invisible

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Interactive conditioned Flow Matching visualizer (Dash).')
    parser.add_argument('--checkpoint-dir', default='checkpoints_fm_cond')
    parser.add_argument('--ode-steps', type=int, default=DEFAULT_ODE_STEPS,
                        help='Euler steps per update (default 30). '
                             'Lower = faster but less accurate.')
    parser.add_argument('--n-points', type=int, default=DEFAULT_N_POINTS)
    parser.add_argument('--port', type=int, default=8050)
    args = parser.parse_args()

    global _model, _x0, _qmasks, _ode_runner
    _model      = load_model(args.checkpoint_dir)
    _ode_runner = make_ode_runner(_model)

    # Pre-generate the maximum number of points; sliders subset this at render time
    tf.random.set_seed(42)
    _x0 = tf.random.normal((MAX_N_POINTS, 2)).numpy()

    qx      = _x0[:, 0] < 0
    qy      = _x0[:, 1] < 0
    _qmasks = {(bx, by): (qx == bx) & (qy == by)
               for bx in (True, False) for by in (True, False)}

    # Warm up: trace the ODE runner with default settings so first request is instant
    print("Warming up ODE runner (tracing tf.function)...")
    _dummy_obs = tf.zeros((DEFAULT_N_POINTS, settings['obs_dim']), dtype=tf.float32)
    _dummy_steps = tf.constant([DEFAULT_ODE_STEPS // (len(SNAP_T_VALUES) - 1)] *
                               (len(SNAP_T_VALUES) - 1), dtype=tf.int32)
    _ode_runner(tf.constant(_x0[:DEFAULT_N_POINTS]),
                _dummy_obs,
                tf.constant(SNAP_T_VALUES, dtype=tf.float32),
                _dummy_steps)
    print("Ready — open http://127.0.0.1:{} in your browser.".format(args.port))

    app = make_app()
    app.run(debug=True, port=args.port)


if __name__ == '__main__':
    main()
