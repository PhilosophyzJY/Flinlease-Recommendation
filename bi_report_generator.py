import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Force the renderer to output standard JSON to prevent browser rendering issues.
pio.renderers.default = "json"

# Tableau-inspired color palette
TABLEAU_COLORS = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
    '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'
]

HTML_TEMPLATE = """
<html>
<head>
    <title>综合商业智能分析报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Noto Serif SC', serif;
            font-weight: 400;
            margin: 0;
            background-color: #f0f2f5;
            color: #3a3a3a;
        }}
        .header {{
            background-color: #ffffff;
            padding: 16px 32px;
            border-bottom: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 700;
            color: #2c3e50;
        }}
        .container {{
            max-width: 1600px;
            margin: 20px auto;
            padding: 0 20px;
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }}
        .grid-container-full {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }}
        .card {{
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 18px;
            font-weight: 700;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        h1.section-title {{
            font-size: 28px;
            font-weight: 700;
            margin-top: 40px;
            margin-bottom: 20px;
            color: #172B4D;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>综合商业智能分析报告 (Comprehensive BI Report)</h1>
    </div>
    <div class="container">
        {body_content}
    </div>
</body>
</html>
"""

def _create_chart_card(title, fig_html):
    """Creates an HTML card for a chart."""
    return f'<div class="card"><h2>{title}</h2>{fig_html}</div>'

def _update_fig_layout(fig, title):
    """Applies standard layout updates to a Plotly figure."""
    fig.update_layout(
        template='plotly_white',
        title=dict(text=title, x=0.05, xanchor='left', font=dict(size=18, weight='bold')),
        font=dict(family="'Noto Serif SC', serif", size=12, color="#3a3a3a"),
        margin=dict(l=50, r=50, b=50, t=70)
    )
    return fig

def generate_overview_figures(df):
    """Generates figures for the overview section."""
    figs = {}

    # Timeseries
    df_ts = df.copy()
    df_ts['year_month'] = df_ts['transaction_date'].dt.to_period('M').astype(str)
    monthly_counts = df_ts.groupby('year_month').size().reset_index(name='count')
    fig_ts = px.line(monthly_counts, x='year_month', y='count', color_discrete_sequence=[TABLEAU_COLORS[0]])
    figs['交易数量时序图 (Transaction Volume Over Time)'] = fig_ts

    # Value Distribution
    df_filtered = df[df['财产价值（万元）'] > 0]
    fig_val = px.histogram(df_filtered, x='财产价值（万元）', log_x=True, color_discrete_sequence=[TABLEAU_COLORS[1]])
    figs['财产价值分布图 (Property Value Distribution)'] = fig_val

    return figs

def generate_geo_figures(df):
    """Generates figures for geographical analysis."""
    figs = {}

    # By Value
    province_agg_val = df.groupby('省份')['财产价值（万元）'].sum().reset_index()
    fig_val = px.bar(province_agg_val, x='省份', y='财产价值（万元）', color_discrete_sequence=[TABLEAU_COLORS[2]])
    figs['各省份业务总价值 (Total Business Value by Province)'] = fig_val

    # By Count
    province_agg_count = df.groupby('省份').size().reset_index(name='count')
    fig_count = px.bar(province_agg_count, x='省份', y='count', color_discrete_sequence=[TABLEAU_COLORS[3]])
    figs['各省份业务总数量 (Total Transaction Count by Province)'] = fig_count

    return figs

def generate_lessee_char_figures(df):
    """Generates figures for lessee characteristics."""
    figs = {}
    pie_chart_dims = {
        '承租人行业分布 (Lessee Industry Distribution)': '申万行业一级',
        '融资期限分布 (Term Bin Distribution)': '期限分箱',
        '承租人地域分布 (Lessee Province Distribution)': '省份',
    }
    for title, dim_col in pie_chart_dims.items():
        if dim_col in df.columns:
            counts = df[dim_col].value_counts()
            fig = px.pie(counts, values=counts.values, names=counts.index, hole=.4, color_discrete_sequence=TABLEAU_COLORS)
            figs[title] = fig
    return figs

def generate_lessor_pref_figures(df, top_n=10):
    """Generates Sankey diagrams for lessor preferences."""
    figs = {}
    top_lessors_by_value = df.groupby('出租人')['财产价值（万元）'].sum().nlargest(top_n).index
    df_top = df[df['出租人'].isin(top_lessors_by_value)]

    preference_dims = {
        '地域偏好 (Province)': '省份', '行业偏好 (Industry)': '申万行业一级',
        '财产价值偏好 (Value Bin)': '价值分箱', '期限偏好 (Term Bin)': '期限分箱'
    }
    for title_prefix, dim_col in preference_dims.items():
        title = f"Top {top_n} 出租人 - {title_prefix} (按价值)"
        agg_data = df_top.groupby(['出租人', dim_col])['财产价值（万元）'].sum().reset_index()
        agg_data.columns = ['source', 'target', 'value']
        all_nodes = pd.concat([agg_data['source'], agg_data['target']]).unique()
        node_map = {name: i for i, name in enumerate(all_nodes)}

        link_colors = [TABLEAU_COLORS[i % len(TABLEAU_COLORS)] for i in agg_data['source'].map(node_map)]

        links = {'source': agg_data['source'].map(node_map), 'target': agg_data['target'].map(node_map), 'value': agg_data['value'], 'color': link_colors}
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=25, thickness=20, line=dict(color="black", width=0.5), label=all_nodes),
            link=links
        )])
        figs[title] = fig

    return figs

def generate_bi_report(df, output_html_path):
    """Generates a comprehensive BI-style HTML report with a Tableau-inspired design."""
    print("\n--- Generating New Tableau-Style BI Report ---")

    all_figures = {
        **generate_overview_figures(df),
        **generate_geo_figures(df),
        **generate_lessee_char_figures(df),
        **generate_lessor_pref_figures(df, top_n=10)
    }

    body_content = ""
    layout = {
        "概览 (Overview)": ["交易数量时序图 (Transaction Volume Over Time)", "财产价值分布图 (Property Value Distribution)"],
        "地域分布分析 (Geographical Distribution)": ["各省份业务总价值 (Total Business Value by Province)", "各省份业务总数量 (Total Transaction Count by Province)"],
        "承租人特征分析 (Lessee Characteristics)": list(generate_lessee_char_figures(df).keys()),
        "出租人偏好分析 (Lessor Preferences)": list(generate_lessor_pref_figures(df).keys())
    }

    for section_title, fig_titles in layout.items():
        section_html = f"<h1 class='section-title'>{section_title}</h1>"
        grid_class = "grid-container-full" if "出租人" in section_title else "grid-container"
        section_html += f'<div class="{grid_class}">'

        for fig_title in fig_titles:
            if fig_title in all_figures:
                fig = all_figures[fig_title]
                # Pass the user-facing title to the layout update function
                _update_fig_layout(fig, fig_title)
                fig_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})
                section_html += _create_chart_card(fig_title, fig_html)

        section_html += "</div>"
        body_content += section_html

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE.format(body_content=body_content))

    print(f"New BI report saved to '{output_html_path}'")
