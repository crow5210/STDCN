import pandas as pd
import plotly.express as px
import plotly.offline as offline
import os

def result_decode(result,out_dir=None):
    report_df = pd.DataFrame({})
    for k,v in result.items():
        data,algo = k.split("_")
        report_dict = {"data":[data],"algo":[algo]}
        for key in v._pool.keys():
            report_dict[key]=[f"{v.get_avg(key):.3f}"]

        report_df = pd.concat([report_df,pd.DataFrame(report_dict)])

    if out_dir!= None:
        report_dir = f"{out_dir}/report"
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_df.to_csv(f"{report_dir}/result.csv")
        
    report_df[report_df.columns[2:]] = report_df[report_df.columns[2:]].astype(float)
    return report_df

def gen_ranking_graph(report_df,out_dir,auto_open=False):
    col_mae = "test_MAE_avg" if "test_MAE_avg" in report_df.columns else "test_MAE"
    fig = px.bar(report_df, 
                        x="algo", 
                        y=["test_time",col_mae], 
                        color_discrete_map={
                            "test_time": "RebeccaPurple", col_mae: "lightsalmon"
                            },
                        template="simple_white",
                        facet_col="data",
                        facet_col_wrap = 2,
                        text_auto=".2",
                        )

    fig.update_traces(textfont_size=12, 
                    textangle=0, 
                    textposition="outside", 
                    cliponaxis=False,
                    )

    fig.update_layout(title="Efficiency analysis", 
                    font_family="San Serif",
                    bargap=0.2,
                    titlefont={'size': 24},
                    width=800,
                    height=500,
                    # legend=dict(
                    #             x=0.0,
                    #             y=1.0,
                    #             bgcolor='rgba(255, 255, 255, 0)',
                    #             bordercolor='rgba(255, 255, 255, 0)'
                    #         ),              
                    )

    # fig.show()
    offline.plot(fig, filename=f'{out_dir}/report/ranking.html', auto_open=auto_open)


def gen_polar_graph(report_df,out_dir,auto_open=False):
    fig = px.line_polar(report_df, r='test_time', theta='algo', color="data",line_close=True)

    # fig.show()
    offline.plot(fig, filename=f'{out_dir}/report/polar.html', auto_open=auto_open)
