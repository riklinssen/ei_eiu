
# Even it Up impact measuremennt, stats from quarterly poll perception research
# Rik Linssen - Feb 2020
# geithub respository here: www.github.com/riklinssen/ei_eiu

#############IMPORTS########################
import numpy as np
import pandas as pd
import pathlib
import datetime
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
##########################BOILERPLATE##########
# Oxfam colors
hex_values = ['#E70052',  # rood
              '#F16E22',  # oranje
              '#E43989',  # roze
              '#630235',  # Bordeax
              '#53297D',  # paars
              '#0B9CDA',  # blauw
              '#61A534',  # oxgroen
              '#0C884A'  # donkergroen
              ]


colormap = {
    'Moderne burgerij': '#E70052',
    'Opwaarts mobielen': '#F16E22',
    'Postmaterialisten': '#E43989',
    'Nieuwe conservatieven': '#630235',
    'Traditionele burgerij': '#53297D',
    'Kosmopolieten': '#0B9CDA',
    'Postmoderne hedonisten': '#FBC43A',
    'Gemaksgeoriënteerden': '#BECE45',
}

mentalitytranslator = {'Moderne burgerij': 'Modern mainstream',
                       'Opwaarts mobielen': 'Social climbers',
                       'Postmaterialisten': 'Post-materialists',
                       'Nieuwe conservatieven': 'New conservatives',
                       'Traditionele burgerij': 'Traditionals',
                       'Kosmopolieten': 'Cosmopolitans',
                       'Postmoderne hedonisten': 'Post-modern hedonists',
                       'Gemaksgeoriënteerden': 'Convenience-oriented'}


segmentcolormap_en = dict(zip(
    [v for k, v in mentalitytranslator.items()],
    [v for k, v in colormap.items()]
))

#add total
segmentcolormap_en['Total']='000000'

# generate grouping into warm cold and peripheral audience
mentalityaudiencemap = {'Modern mainstream': 'Peripheral A',
                        'Social climbers': 'Cold',
                        'Post-materialists': 'Warm',
                        'New conservatives': 'Peripheral B',
                        'Traditionals': 'Peripheral A',
                        'Cosmopolitans': 'Warm',
                        'Post-modern hedonists': 'Peripheral B',
                        'Convenience-oriented': 'Cold'
                        }


mentalityaudiencemap_nl={'Moderne burgerij': 'Peripheral A',
                        'Opwaarts mobielen': 'Cold',
                        'Postmaterialisten': 'Warm',
                        'Nieuwe conservatieven': 'Peripheral B',
                        'Traditionele burgerij': 'Peripheral A',
                        'Kosmopolieten': 'Warm',
                        'Postmoderne hedonisten': 'Peripheral B',
                        'Gemaksgeoriënteerden': 'Cold'
                        }


# audiencecolormap
# colors hot/cold

# hot        #dd6e6e
# lauw       #ed8a38
# cold       #0f2491


# audience colors
audiencecolormap = dict(zip(
    ['Warm', 'Peripheral A',
        'Peripheral B', 'Cold'],
    ['#dd6e6e', '#ed8a38', '#eecd75', '#0f2491']
))

# weight funcs


def wavg_func(datacol, weightscol):
    def wavg(group):
        dd = group[datacol]
        ww = group[weightscol] * 1.0
        return (dd * ww).sum() / ww.sum()
    return wavg




def df_wavg(df, groupbycol, weightscol):
    grouped = df.groupby(groupbycol)
    df_ret = grouped.agg({weightscol: sum})
    datacols = [cc for cc in df.columns if cc not in [groupbycol, weightscol]]
    for dcol in datacols:
        try:
            wavg_f = wavg_func(dcol, weightscol)
            df_ret[dcol] = grouped.apply(wavg_f)
        except TypeError:  # handle non-numeric columns
            df_ret[dcol] = grouped.agg({dcol: min})
    return df_ret




def grouped_weights_statscol (df, statscol, groupbycol, weightscol):
    df.dropna(subset=[statscol], inplace=True)
    nrobs=len(df)
    grouped=df.groupby(groupbycol)
    stats={}
    means=[]
    lower=[]
    upper=[]
    groups=list(grouped.groups.keys())
    for gr in groups:
        stats=DescrStatsW(grouped.get_group(gr)[statscol], weights=grouped.get_group(gr)[weightscol], ddof=0)
        means.append(stats.mean)
        lower.append(stats.tconfint_mean()[0])
        upper.append(stats.tconfint_mean()[1])
    weightedstats=pd.DataFrame([means, lower, upper], columns=groups, index=['weighted mean', 'lower bound', 'upper bound']).T
    weightedstats['numberofobs']=nrobs
    return weightedstats

    #weightedstats=pd.DataFrame([means, lower, upper], index=groups)
    #return weightedstats





##try to generalize func towards more cols
def grouped_weights_statsdf(df, statscols, groupbycol, weightscol):
    """generates df with weighted means and 95% CI by groupbycol for cols in statscols
    

    Parameters
    ----------
    df : df
        df to be weigthed
    statscols : list
        cols/outcomes for weigthed stats
    groupbycol : str
        column name in df that defines groups 
    weightscol : str
        column name in df with weigths 
              
    
    Returns
    -------
    df
        multi-indexed df with outcome and groups as index
        stats generated: weighted mean, upper bound (95 CI), lower bound (95% CI), weighted n by group, total n unweighted

    """    
    alldata=pd.DataFrame()
    for c in statscols: 
        cdf=df.dropna(subset=[c])
        nrobs=len(cdf)
        grouped=cdf.groupby(groupbycol)
        stats={}
        means=[]
        lower=[]
        upper=[]
        nrobs_gr=[]
        groups=list(grouped.groups.keys())
        for gr in groups:
            stats=DescrStatsW(grouped.get_group(gr)[c], weights=grouped.get_group(gr)[weightscol], ddof=0)
            means.append(stats.mean)
            lower.append(stats.tconfint_mean()[0])
            upper.append(stats.tconfint_mean()[1])
            nrobs_gr.append(stats.nobs)          
        weightedstats=pd.DataFrame([means, lower, upper, nrobs_gr], columns=groups, index=['weighted mean', 'lower bound', 'upper bound', 'wei_n__group']).T
        weightedstats['tot_n_unweigthed']=nrobs
        weightedstats['outcome']=c
        weightedstats.index.name='groups'
        colstats=weightedstats.reset_index()
        colstats=colstats.set_index(['outcome', 'groups'])
        alldata=pd.concat([alldata, colstats])
               
    return alldata

def autolabelpercentmid(ax, xpos='center'):
    """
    Attach a text label above each bar (a percentage) in *ax*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    bars=ax.patches
    for bar in bars:
        height = bar.get_height()
        heightval=str(int(bar.get_height()*100))+ '%'
        ax.text(bar.get_x() + bar.get_width()*offset[xpos], (0.5*height),
        heightval, fontsize=8, ha=ha[xpos], va='bottom', color='white', alpha=1)


def autolabelpercenttop(ax, xpos='center'):
    """
    Attach a text label above each bar (a percentage) in *ax*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    bars=ax.patches
    for bar in bars:
        height = bar.get_height()
        heightval=str(int(bar.get_height()*100))+ '%'
        ax.text(bar.get_x() + bar.get_width()*offset[xpos], (1.01*height),
        heightval, fontsize=12, ha=ha[xpos], va='bottom', color=bar.get_facecolor(), alpha=1)






##FILEPATHS

base_path = pathlib.Path.cwd()
data = base_path / "data"
graphs = base_path / "graphs"



#MAKE DATASETS
q32019= pd.read_pickle(data/"q3_2019_clean.pkl" )


# check data


colslist=[ 'eiu_heard_taxevasion_heard',
 'eiu_heard_taxevasion_know',
 'eiu_heard_taxevasion_on_heard',
 'eiu_heard_taxevasion_on_know',
 'eiu_heard_taxevasion_on_petition_seen',
 'eiu_heard_taxevasion_on_dolfj_seen',
 'mentality',
 'wgprop']
eiu_df=(q32019.loc[:,colslist]
    .assign(total=lambda x: 'Total') #add 1 for totals
    .assign(mentality_eng=lambda x: x['mentality'].map(mentalitytranslator)) # add mentality in eng
    )  
#heard but did not know about. 
eiu_df['eiu_heard_taxevasion_heard_notknow']=np.where((eiu_df['eiu_heard_taxevasion_heard']==1) & (eiu_df['eiu_heard_taxevasion_know']==0), 1, 0 )
#for on re tax evasion
eiu_df['eiu_heard_taxevasion_on_heard_notknow']=np.where((eiu_df['eiu_heard_taxevasion_on_heard']==1) & (eiu_df['eiu_heard_taxevasion_on_know']==0), 1, 0 )

#set nans in petition and dolfj to 0 for whole population

for c in ['eiu_heard_taxevasion_on_petition_seen', 'eiu_heard_taxevasion_on_dolfj_seen']: 
    print(eiu_df[c].value_counts(dropna=False))
    eiu_df[c]=eiu_df[c].fillna(0)
    print(eiu_df[c].value_counts(dropna=False))

 


outputcols=['eiu_heard_taxevasion_heard',
    'eiu_heard_taxevasion_heard_notknow',
    'eiu_heard_taxevasion_know',
    'eiu_heard_taxevasion_on_heard',
    'eiu_heard_taxevasion_on_heard_notknow',
    'eiu_heard_taxevasion_on_know',
    'eiu_heard_taxevasion_on_petition_seen',
    'eiu_heard_taxevasion_on_dolfj_seen']
for c in outputcols: 
    print(eiu_df[c].value_counts(dropna=False))

for c in outputcols: 
    print(eiu_df[c].value_counts(dropna=False, normalize=True))

eiu_stats_by_total=grouped_weights_statsdf(eiu_df, outputcols, 'total', 'wgprop')

eiu_stats_by_mt=grouped_weights_statsdf(eiu_df, outputcols, 'mentality_eng', 'wgprop')

#make indexslice 
idx = pd.IndexSlice

# sel_t=eiu_stats_by_total.loc[idx['eiu_heard_taxevasion_heard', 'eiu_heard_taxevasion_know', 'eiu_heard_taxevasion_heard_notknow'],:,:]
# sel_mt=eiu_stats_by_mt.loc[idx['eiu_heard_taxevasion_heard', 'eiu_heard_taxevasion_know', 'eiu_heard_taxevasion_heard_notknow'],:,:].sort_values(by='weighted mean')



sel_t=eiu_stats_by_total.loc[idx['eiu_heard_taxevasion_heard',:,:]]
sel_mt=eiu_stats_by_mt.loc[idx['eiu_heard_taxevasion_heard',:,:]].sort_values(by='weighted mean', ascending=False)
sortorder={v: i for i, v in enumerate(sel_mt.index)}

#add colors
sel_t['color']=sel_t.index.map(segmentcolormap_en)
sel_mt['color']=sel_mt.index.map(segmentcolormap_en)

#errorbars
#add errorterms for easy plotting 
sel_t['err']=sel_t['upper bound']-sel_t['weighted mean']
sel_mt['err']=sel_mt['upper bound']-sel_mt['weighted mean']




# EiUA1. Heb je in de afgelopen 12 maanden in het nieuws of op sociale media iets gezien of gehoord over belastingontwijking door grote bedrijven?

filename=graphs/'heardabouttaxev.svg'
widths=[1,6]
heights=[1]
fig=plt.figure(figsize=(10.48, 6.55))
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=widths, height_ratios=heights)
ax1=fig.add_subplot(gs[0, 0])
ax1.bar(x=sel_t.index, height=sel_t['weighted mean'],  color=sel_t['color'], yerr=sel_t['err'], ecolor='lightgrey')
ax1.set_title('Total \n(% of population)')
ax2=fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.bar(x=sel_mt.index, height=sel_mt['weighted mean'], color=sel_mt['color'], yerr=sel_mt['err'], ecolor='lightgrey')
ax2.set_title('by segment \n(% of segment)')
for ax in fig.axes:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0,1)
    ax.set_ylabel('% of people that heard about tax evasion', fontstyle='italic') 
    autolabelpercenttop(ax, xpos='left')

    #labels
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha('center')
        label.set_fontsize('large')
    
    #spines
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax2.axes.get_yaxis().set_visible(False)
ax2.spines['left'].set_visible(False)

fig.suptitle("Has heard about tax evasion in the news", size=14, y=1.05, color='black')

#footnotes
nrobs=str(sel_t.at['Total','tot_n_unweigthed'])
plt.figtext(0, -0.2, 'Source: Quarterly poll Q3 2019 (Sept),' + ' n=' +nrobs +'\nvertical lines represent 95% confidence intervals' ,  size='small')


fig.savefig(filename,  facecolor='w', bbox_inches='tight')



### has heard and knows what it was about detailed breakdown

##make sortorder for segments equal to previous graphs


sel_notknow_t=eiu_stats_by_total.loc[idx[['eiu_heard_taxevasion_heard_notknow'],:,:]].droplevel(level=0)
sel_know_t=eiu_stats_by_total.loc[idx[['eiu_heard_taxevasion_know'],:,:]].droplevel(level=0)


sel_notknow_mt=eiu_stats_by_mt.loc[idx[['eiu_heard_taxevasion_heard_notknow'],:,:]].droplevel(level=0)
sel_notknow_mt['order']=sel_notknow_mt.index.map(sortorder)
sel_notknow_mt=sel_notknow_mt.sort_values(by='order')

sel_know_mt=eiu_stats_by_mt.loc[idx[['eiu_heard_taxevasion_know'],:,:]].droplevel(level=0)
sel_know_mt['order']=sel_know_mt.index.map(sortorder)
sel_know_mt=sel_know_mt.sort_values(by='order')



#add colors
sel_notknow_t['color']=sel_notknow_t.index.map(segmentcolormap_en)
sel_know_t['color']=sel_know_t.index.map(segmentcolormap_en)


sel_notknow_mt['color']=sel_notknow_mt.index.map(segmentcolormap_en)
sel_know_mt['color']=sel_know_mt.index.map(segmentcolormap_en)




filename=graphs/'heardabouttaxev_detail.svg'
widths=[1,6]
heights=[1]
fig=plt.figure(figsize=(10.48, 6.55))
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=widths, height_ratios=heights)
ax1=fig.add_subplot(gs[0, 0])
ax1.bar(x=sel_notknow_t.index, height=sel_notknow_t['weighted mean'],  color=sel_notknow_t['color'], alpha=0.6)
ax1.bar(x=sel_know_t.index, height=sel_know_t['weighted mean'], color=sel_know_t['color'], bottom=sel_notknow_t['weighted mean'])
#texts
ax1.text(x=sel_notknow_t.index,y=(sel_notknow_t['weighted mean']/2), s=str(int(sel_notknow_t['weighted mean']*100))+ '%', color='white',ha='center')
ax1.text(x=sel_know_t.index,y=(sel_notknow_t['weighted mean']+(sel_know_t['weighted mean']/2)), s=str(int(sel_know_t['weighted mean']*100))+ '%', color='white', ha='center')
ax1.text(x=sel_know_t.index,y=(sel_notknow_t['weighted mean']+sel_know_t['weighted mean'])*1.05, s=str(int((sel_notknow_t['weighted mean']+sel_know_t['weighted mean'])*100))+ '%', color='black', ha='center')
ax1.set_title('Total \n(% of population)')

#by mentality
ax2=fig.add_subplot(gs[0, 1], sharey=ax1)
b1=ax2.bar(x=sel_notknow_mt.index, height=sel_notknow_mt['weighted mean'], color=sel_notknow_mt['color'], alpha=0.6)
for bar in b1.patches:
    height = bar.get_height()
    heightval=str(int(bar.get_height()*100))+ '%'
    ax2.text((bar.get_x() + bar.get_width()/2), (0.5*height), heightval, fontsize=8, ha='center', color='white')

b2=ax2.bar(x=sel_know_mt.index, height=sel_know_mt['weighted mean'], color=sel_know_mt['color'], bottom=sel_notknow_mt['weighted mean'])
for bar in b2.patches:
    bottom =bar.get_y()
    height = bar.get_height()
    heightval=str(int(bar.get_height()*100))+ '%'
    #middle
    ax2.text((bar.get_x() + bar.get_width()/2), (bottom+(0.5*height)), heightval, fontsize=12, ha='center', color='white')
    #total
    heighttop=(bottom+height)
    heightvaltop=str(int(heighttop*100))+ '%'
    ax2.text((bar.get_x() + bar.get_width()/2), heighttop*1.05, heightvaltop,  ha='center', color=bar.get_facecolor(), fontsize=12)
ax2.set_title('by segment \n(% of segment)')

for ax in fig.axes:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0,1)
    ax.set_ylabel('% of people', fontstyle='italic') 
    

    #labels
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha('center')
        label.set_fontsize('large')
    
    #spines
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax2.axes.get_yaxis().set_visible(False)
ax2.spines['left'].set_visible(False)

fig.suptitle("Has heard about tax evasion in the news, \n and (does not) remember what that was about ", size=14, y=1.05, color='black')

#footnotes
nrobs=str(sel_t.at['Total','tot_n_unweigthed'])
plt.figtext(0, -0.25, 'Source: Quarterly poll Q3 2019 (Sept),' + ' n=' +nrobs+'.\nTransparent bars represent share of people who have heard about tax evasion but cannot remember exactly what that was about.\nDark bars represent people who have heard about it and know exactly what that was about. \nCumulative percentages plotted on top of bars.'  ,  size='small')


fig.savefig(filename,  facecolor='w', bbox_inches='tight')





### has heard about tax evasion from ON and knows what it was about detailed breakdown



sel_notknow_on_t=eiu_stats_by_total.loc[idx[['eiu_heard_taxevasion_on_heard_notknow'],:,:]].droplevel(level=0)
sel_know_on_t=eiu_stats_by_total.loc[idx[['eiu_heard_taxevasion_on_know'],:,:]].droplevel(level=0)




#add colors
sel_notknow_on_t['color']=sel_notknow_t.index.map(segmentcolormap_en)
sel_know_on_t['color']=sel_know_t.index.map(segmentcolormap_en)

#further breakdown by segment is not possible sel_know_on_t nrobs<105


filename=graphs/'heard_on_abouttaxev_total.svg'

fig=plt.figure(figsize=((10.48/2), 6.55))
ax1=fig.add_subplot()
ax1.bar(x=sel_notknow_on_t.index, height=sel_notknow_on_t['weighted mean'],  color='#61A534', alpha=0.6)
ax1.bar(x=sel_know_on_t.index, height=sel_know_on_t['weighted mean'], color='#0C884A', bottom=sel_notknow_on_t['weighted mean'])
#texts
ax1.text(x=sel_notknow_on_t.index,y=(sel_notknow_on_t['weighted mean']/2), s=str(int(sel_notknow_on_t['weighted mean']*100))+ '%', color='white',ha='center')
ax1.text(x=sel_know_on_t.index,y=(sel_notknow_on_t['weighted mean']+sel_know_on_t['weighted mean'])*1.1, s=str(int((sel_notknow_on_t['weighted mean']+sel_know_on_t['weighted mean'])*100))+ '%', color='black', ha='center')
ax1.set_title('Total \n(% of population)')


for ax in fig.axes:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0,1)
    ax.set_ylabel('% of people', fontstyle='italic') 
    

    #labels
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha('center')
        label.set_fontsize('large')
    
    #spines
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle("Has heard Oxfam Novib \nabout tax evasion in the news, \n and (does not) remember what that was about ", size=14, y=1.05, color='black')

#footnotes
nrobs=str(sel_t.at['Total','tot_n_unweigthed'])
plt.figtext(0, -0.1, 'Source: Quarterly poll Q3 2019 (Sept),' + ' n=' +nrobs+'.\nLight green bar represent share of people who have heard about tax evasion \nbut cannot remember exactly what that was about.\nDark green bars represent people who have heard about it\nand know exactly what that was about (1.6%). \nCumulative percentages plotted on top of bars.'  ,  size='small')


fig.savefig(filename,  facecolor='w', bbox_inches='tight')
fig.show()



##call for petition and dolf jansen

#select data
sel_petition_t=eiu_stats_by_total.loc[idx[['eiu_heard_taxevasion_on_petition_seen'],:,:]].droplevel(level=0)
sel_dolf_t=eiu_stats_by_total.loc[idx[['eiu_heard_taxevasion_on_dolfj_seen'],:,:]].droplevel(level=0)

#errorbars
#add errorterms for easy plotting 
sel_petition_t['err']=sel_petition_t['upper bound']-sel_petition_t['weighted mean']
sel_dolf_t['err']=sel_dolf_t['upper bound']-sel_dolf_t['weighted mean']

filename=graphs/'seen_dolf_petition.svg'
fig=plt.figure(figsize=((10.48), 6.55))
gs = fig.add_gridspec(nrows=1, ncols=2)

ax1=fig.add_subplot(gs[0,0])
ax1.bar(x=sel_petition_t.index, height=sel_petition_t['weighted mean'], color='#697093', yerr=sel_petition_t['err'], ecolor='lightgrey')
ax1.set_title('Seen blue envelope \n(% of population)')

ax2=fig.add_subplot(gs[0,1], sharey=ax1)
ax2.bar(x=sel_dolf_t.index, height=sel_dolf_t['weighted mean'], color='#0C884A',  yerr=sel_dolf_t['err'], ecolor='lightgrey')
ax2.set_title('Seen Dolf Jansen\n(% of population)')


for ax in fig.axes:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0,1)
    ax.set_ylabel('% of people that have seen message', fontstyle='italic') 
    autolabelpercenttop(ax, xpos='left')

    #labels
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha('center')
        label.set_fontsize('large')
    
    #spines
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax2.axes.get_yaxis().set_visible(False)
ax2.spines['left'].set_visible(False)

fig.suptitle("Seen messages: \nblue envelope & Dolf Jansen", size=14, y=1.05, color='black')

#footnotes
nrobs=str(sel_t.at['Total','tot_n_unweigthed'])
plt.figtext(0, -0.1, 'Source: Quarterly poll Q3 2019 (Sept),' + ' n=' +nrobs +'\nvertical lines represent 95% confidence intervals' ,  size='small')

fig.savefig(filename,  facecolor='w', bbox_inches='tight')


fig.show()
