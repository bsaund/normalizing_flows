#!/usr/bin/env python
"""
Interactive text-to-points preview.

Type any word and see side-by-side:
  Left  — the rendered text image (what gets fed to the point sampler)
  Right — the 2-D scatter of sampled points (the actual training data)

Sliders let you tune font size and number of points in real time.

Usage:
    python text_preview.py
    Then open http://127.0.0.1:8051 in your browser.
"""
from __future__ import print_function

import base64
import io

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

from generate_points import render_text_image, create_points_from_text

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TEXT       = 'HELLO'
DEFAULT_FONT_SIZE  = 120
DEFAULT_NUM_POINTS = 5000
POINT_LIM          = 3.5     # axis limits for the scatter plot

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title='Text Preview')

_label = lambda text, color: html.Label(
    text, style={'fontWeight': 'bold', 'fontSize': '13px', 'color': color})

app.layout = html.Div(
    style={'fontFamily': 'sans-serif', 'padding': '24px',
           'backgroundColor': '#ffffff', 'maxWidth': '1200px', 'margin': '0 auto'},
    children=[
        html.H2('Text → Points Preview',
                style={'color': '#333344', 'marginBottom': '4px'}),
        html.P('Type a word, adjust font size and point count, see the training data.',
               style={'color': '#888888', 'marginBottom': '20px'}),

        # ── Controls ──────────────────────────────────────────────────────
        html.Div(
            style={'display': 'flex', 'gap': '32px', 'alignItems': 'flex-end',
                   'marginBottom': '24px', 'flexWrap': 'wrap'},
            children=[
                # Text input
                html.Div(children=[
                    _label('Word', '#333344'),
                    dcc.Input(
                        id='text-input',
                        type='text',
                        value=DEFAULT_TEXT,
                        debounce=True,          # fires on Enter or blur
                        maxLength=10,
                        style={
                            'fontSize': '28px',
                            'fontWeight': 'bold',
                            'width': '180px',
                            'padding': '6px 10px',
                            'border': '2px solid #4455cc',
                            'borderRadius': '6px',
                            'textTransform': 'uppercase',
                            'letterSpacing': '4px',
                        },
                    ),
                ]),

                # Font size slider
                html.Div(style={'flex': 1, 'minWidth': '200px'}, children=[
                    _label('Font size', '#557755'),
                    dcc.Slider(
                        id='slider-font',
                        min=40, max=220, step=10,
                        value=DEFAULT_FONT_SIZE,
                        marks={v: str(v) for v in [40, 80, 120, 160, 220]},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='mouseup',
                    ),
                ]),

                # Num points slider
                html.Div(style={'flex': 1, 'minWidth': '200px'}, children=[
                    _label('Num points', '#885599'),
                    dcc.Slider(
                        id='slider-npts',
                        min=500, max=20000, step=500,
                        value=DEFAULT_NUM_POINTS,
                        marks={v: str(v) for v in [500, 5000, 10000, 20000]},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='mouseup',
                    ),
                ]),
            ],
        ),

        # ── Loading wrapper (spinner in top-right corner) ─────────────────
        html.Div(
            style={'position': 'relative'},
            children=[
                # ── Display panels ────────────────────────────────────────
                html.Div(
                    style={'display': 'flex', 'gap': '24px', 'alignItems': 'flex-start'},
                    children=[
                        # Left: rendered image
                        html.Div(
                            style={'flex': 1},
                            children=[
                                html.H4('Rendered image',
                                        style={'color': '#555577', 'marginBottom': '8px'}),
                                html.Img(
                                    id='text-image',
                                    style={
                                        'width': '100%',
                                        'border': '1px solid #dddddd',
                                        'borderRadius': '4px',
                                        'imageRendering': 'crisp-edges',
                                    },
                                ),
                                html.Div(id='image-info',
                                         style={'fontSize': '12px', 'color': '#999999',
                                                'marginTop': '4px'}),
                            ],
                        ),

                        # Right: sampled points
                        html.Div(
                            style={'flex': 1},
                            children=[
                                html.H4('Sampled points',
                                        style={'color': '#555577', 'marginBottom': '8px'}),
                                dcc.Loading(
                                    type='circle', color='#4455cc',
                                    children=dcc.Graph(
                                        id='points-plot',
                                        config={'displayModeBar': False},
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

@app.callback(
    Output('text-image',  'src'),
    Output('image-info',  'children'),
    Output('points-plot', 'figure'),
    Input('text-input',   'value'),
    Input('slider-font',  'value'),
    Input('slider-npts',  'value'),
)
def update(text, font_size, num_points):
    text       = (text or DEFAULT_TEXT).upper()
    font_size  = int(font_size  or DEFAULT_FONT_SIZE)
    num_points = int(num_points or DEFAULT_NUM_POINTS)

    # ── Render image ──────────────────────────────────────────────────────
    img = render_text_image(text, font_size=font_size)
    w, h = img.size

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    img_src = 'data:image/png;base64,' + b64
    info = '{}×{} px'.format(w, h)

    # ── Sample points ─────────────────────────────────────────────────────
    pts = create_points_from_text(text, num_points, font_size=font_size)

    fig = go.Figure(go.Scattergl(
        x=pts[:, 0].tolist(),
        y=pts[:, 1].tolist(),
        mode='markers',
        marker=dict(size=2, color='#3355cc', opacity=0.45),
        name='points',
    ))
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(range=[-POINT_LIM, POINT_LIM], showgrid=True,
                   zeroline=True, zerolinecolor='#aaaaaa',
                   title='x', fixedrange=True),
        yaxis=dict(range=[-POINT_LIM, POINT_LIM], showgrid=True,
                   zeroline=True, zerolinecolor='#aaaaaa',
                   scaleanchor='x', scaleratio=1,
                   title='y', fixedrange=True),
        height=420,
        margin=dict(l=40, r=20, t=20, b=40),
        title=dict(
            text='"{}"  —  {:,} pts  |  font size {}'.format(text, num_points, font_size),
            font=dict(size=12, color='#555577'),
            x=0.5,
        ),
    )

    return img_src, info, fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Open http://127.0.0.1:8051 in your browser.")
    app.run(debug=True, port=8051)
