import matplotlib.pyplot as plt 

plt.style.use('default')

def plot_E_inits(dict_all_E_inits):
    E_inits=list(dict_all_E_inits.keys())
    fig,ax=plt.subplots()
    x1, x2, y1, y2 = -1, 1000, -0.2, 0.5  # subregion of the original image
    axins = ax.inset_axes(
    [0.3, 0.3, 0.67, 0.67],
    xlim=(x1, x2), ylim=(y1, y2), )
    for i in E_inits[:-2]:
        ax.plot(dict_all_E_inits[i],alpha=0.79,label=f"Init E value: {i}")
        ax.axhline(dict_all_E_inits,linestyle="-.",alpha=0.8,color="red",linewidth=0.3)
        ax.set_ylabel("E_init")
        ax.set_xlabel("Epochs")
        axins.plot(dict_all_E_inits[i])
        axins.axhline(i["E_real"],linestyle="-.",alpha=0.7,color="red",linewidth=1)

    plt.title("E_true: 0.032")
    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.tight_layout()
    return fig


    #plt.legend()