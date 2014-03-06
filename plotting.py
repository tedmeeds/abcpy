import pylab as pp

def set_tick_fonsize( sp, fs ):
  for tick in sp.yaxis.get_major_ticks():
    tick.label.set_fontsize(fs)

  for tick in sp.xaxis.get_major_ticks():
    tick.label.set_fontsize(fs)

def set_title_fonsize( sp, ts ):
  sp.title.set_fontsize(ts)
  
def set_label_fonsize( sp, ls ):
  sp.xaxis.label.set_fontsize(ls)
  sp.yaxis.label.set_fontsize(ls)