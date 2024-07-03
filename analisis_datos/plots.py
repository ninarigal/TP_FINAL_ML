from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_histogram(precios):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    ax[0].hist(precios, bins=100, alpha=0.6, color='b', density=True)
    ax[0].set_xlabel('Precio')
    ax[0].set_ylabel('Frecuencia')

    mu, std = norm.fit(precios)
    xmin, xmax = ax[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax[0].plot(x, p, 'k', linewidth=2)


    ax[1].hist(np.log1p(precios), bins=100, alpha=0.6, color='b', density=True)
    ax[1].set_xlabel(r'log[Precio]')
    ax[1].set_ylabel(r'Frecuencia')

    mu, std = norm.fit(np.log1p(precios))
    xmin, xmax = ax[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax[1].plot(x, p, 'k', linewidth=2)


    ax[2].hist(np.sqrt(precios), bins=100, alpha=0.6, color='b', density=True)
    ax[2].set_xlabel(r'$\sqrt{Precio}$')
    ax[2].set_ylabel(r'Frecuencia')

    mu, std = norm.fit(np.sqrt(precios))
    xmin, xmax = ax[2].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax[2].plot(x, p, 'k', linewidth=2)

    plt.show()


def plot_correlaciones(precio, features):

    print('SHAPE:', features.shape)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
    axes = axes.flatten()
    columnas_numeric = features.columns
    for i, colum in enumerate(columnas_numeric):
        x = features[colum]
        y = precio
        axes[i].scatter(x, y, color='b', alpha=0.4, marker='.')
        axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
        axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
        axes[i].tick_params(labelsize=6)
        axes[i].set_xlabel(colum)
        axes[i].set_ylabel(r'Precio')
        if colum == 'Kilómetros':
            axes[i].set_xlim(-10, 800000)
        if colum == 'Edad':
            axes[i].set_xlim(-10, 40)
    fig.tight_layout()
    plt.show()



def plot_cobertura(gama1, gama2, gama3):
    gamas = [gama1, gama2, gama3]

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    for i in range(len(gamas)):
        km = gamas[i]['Kilómetros']	
        edad = gamas[i]['Edad']
        ax[i].scatter(km, edad, label='Gama'+str(i+1), color='b', alpha=0.6, marker='.')
        ax[i].set_xlabel(r'Kilómetros')
        ax[i].set_ylabel(r'Edad')
        ax[i].set_xlim(-10, 800000)
    plt.show()

