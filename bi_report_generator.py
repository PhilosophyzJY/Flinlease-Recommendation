import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def _create_sankey_figure(df, source_col, target_col, agg_col, agg_func, title):
    """Helper function to create a single Sankey figure."""
    if agg_func == 'sum':
        agg_data = df.groupby([source_col, target_col])[agg_col].sum().reset_index()
    else: # count
        agg_data = df.groupby([source_col, target_col])[agg_col].count().reset_index()

    agg_data.columns = ['source', 'target', 'value']

    all_nodes = pd.concat([agg_data['source'], agg_data['target']]).unique()
    node_map = {name: i for i, name in enumerate(all_nodes)}

    links = {
        'source': agg_data['source'].map(node_map),
        'target': agg_data['target'].map(node_map),
        'value': agg_data['value']
    }

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes),
        link=links
    )])
    fig.update_layout(title_text=title, font_size=10)
    return fig


def generate_sankey_figures(df, top_n=20):
    """Generates Plotly Sankey diagram figures for lessor preferences (by value and count)."""
    figs = {}

    top_lessors_by_value = df.groupby('出租人')['财产价值（万元）'].sum().nlargest(top_n).index
    df_top = df[df['出租人'].isin(top_lessors_by_value)]

    preference_dims = {
        '地域偏好 (Province)': '省份',
        '行业偏好 (Industry)': '申万行业一级',
        '财产价值偏好 (Value Bin)': '价值分箱',
        '期限偏好 (Term Bin)': '期限分箱'
    }

    for title_prefix, dim_col in preference_dims.items():
        title_val = f"Top {top_n} 出租人 - {title_prefix} (按价值)"
        fig_val = _create_sankey_figure(df_top, '出租人', dim_col, '财产价值（万元）', 'sum', title_val)
        figs[title_val] = fig_val

        title_count = f"Top {top_n} 出租人 - {title_prefix} (按数量)"
        fig_count = _create_sankey_figure(df_top, '出租人', dim_col, '财产价值（万元）', 'count', title_count)
        figs[title_count] = fig_count

    return figs


def generate_pie_chart_figures(df):
    """Generates Plotly pie chart figures for lessee characteristics."""
    figs = {}
    pie_chart_dims = {
        '承租人行业分布 (Lessee Industry Distribution)': '申万行业一级',
        '融资金额分布 (Value Bin Distribution)': '价值分箱',
        '融资期限分布 (Term Bin Distribution)': '期限分箱',
        '承租人地域分布 (Lessee Province Distribution)': '省份',
    }

    for title, dim_col in pie_chart_dims.items():
        # Check if the column exists to avoid errors
        if dim_col in df.columns:
            counts = df[dim_col].value_counts()
            fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=.3)])
            fig.update_layout(title_text=title)
            figs[title] = fig
    return figs


def generate_china_map_figure(df):
    """Generates a visualization of business distribution by province."""
    try:
        province_agg = df.groupby('省份')['财产价值（万元）'].sum().reset_index()
        province_agg = province_agg.sort_values('财产价值（万元）', ascending=False).head(20)
        fig = px.bar(province_agg, x='省份', y='财产价值（万元）', title='各省份业务分布 (Top 20 Provinces by Business Value)')
        map_note = "<p><b>Note:</b> A map visualization of China could not be generated due to the lack of a required GeoJSON file. A bar chart is provided instead.</p>"
        return fig, map_note
    except Exception as e:
        print(f"Error generating province visualization: {e}")
        return go.Figure(), f"<p>Error generating province visualization: {e}</p>"


def generate_bi_report(df, output_html_path):
    """
    Generates a comprehensive BI-style HTML report with various visualizations.
    """
    print("\n--- Generating Comprehensive BI Report ---")

    province_fig, map_note = generate_china_map_figure(df)
    sankey_figs = generate_sankey_figures(df)
    pie_figs = generate_pie_chart_figures(df)

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write("<html><head><title>综合商业智能分析报告</title>")
        f.write("<style>body {font-family: sans-serif; padding: 20px;} h1, h2, h3 {color: #333;} .plotly-graph-div {margin-bottom: 40px; border: 1px solid #ddd; padding: 10px;}</style>")
        f.write("</head><body>")
        f.write("<h1>综合商业智能分析报告 (Comprehensive BI Report)</h1>")

        f.write("<h2>地域分布分析 (Geographical Distribution Analysis)</h2>")
        f.write(map_note)
        f.write(province_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<hr>")

        f.write("<h2>出租人偏好分析 (Lessor Preference Analysis)</h2>")
        for title, fig in sankey_figs.items():
            f.write(f"<h3>{title}</h3>")
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<hr>")

        f.write("<h2>承租人特征分析 (Lessee Characteristics Analysis)</h2>")
        for title, fig in pie_figs.items():
            f.write(f"<h3>{title}</h3>")
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write("</body></html>")

    print(f"BI report saved to '{output_html_path}'")
