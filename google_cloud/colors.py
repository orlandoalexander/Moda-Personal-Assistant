from scipy import cluster
import numpy as np

def get_colors(img_array):
    clusters = 4
    sub = 0

    shape = img_array.shape
    img_ar = img_array.reshape(-1, shape[2]).astype(float) # reshape image --> 2D vector

    # Kmeans algorithm on 2D vector to form 'num_clusters':
    codebook, dist_mean = cluster.vq.kmeans(img_ar, clusters)
    # 'codebook' stores rgb value for each cluster
    # 'dist_mean' stores mean euclidian distance between each observation and its closest centroid

    # assign code to each observation in image vector 'img_ar' (code value corresponds to the rgb color cluster which is closest to the observation)
    codes, dists = cluster.vq.vq(img_ar, codebook)
    # 'codes' stores code for each observation in 'img_ar' which corresponds to rgb value in 'codebook'
    # 'dists' stores euclidian distance between each observation and its nearest rgb color cluster

    code_counts = sorted({code: np.sum(codes == code) for code in codes}.items(), key=lambda x:x[1],reverse=True)
    color_counts = {code[1]: codebook[code[0]] for code in code_counts[:4]}
    if (np.mean(list(color_counts.values())[0]) -3) < 5:
        sub = list(color_counts.keys())[0]
        color_counts.pop(list(color_counts.keys())[0])
    elif (np.mean(list(color_counts.values())[1]) -3) < 5:
        sub = list(color_counts.keys())[1]
        color_counts.pop(list(color_counts.keys())[1])

    for i,color in enumerate(color_counts.items()):
        rgb = tuple(color[1]/255)
        hex_ = rgb2hex(*color[1].astype('int'))
        perc = round((color[0]/(img_ar.shape[0]-sub)*100),2)
        rectangle = plt.Rectangle((0,0), 100, 100,fc=rgb,fill=True)
        ax = fig.add_subplot(gs[1, i])
        ax.add_patch(rectangle)
        ax.title.set_text(f'Hex: {hex_} | Percentage: {perc}%')
    return
