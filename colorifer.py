import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
import scipy.ndimage

class colorifer():
    def __init__(self, n_colors=8, scale=1, blurrer=ImageFilter.GaussianBlur(3), filter_size=20, n_noise_filter=1):
        self.n_colors = n_colors
        self.scale = scale
        self.clst = KMeans(n_clusters=self.n_colors)
        self.blurrer = blurrer
        self.filter_size=20
        self.n_noise_filter = n_noise_filter

    def fit(self, filename):
        self.img_data = self.load_image(filename)
        img_data_2d = self.img_data.reshape((self.img_data.shape[0]*self.img_data.shape[1], 3))
        self.clst.fit(img_data_2d)
        
        self.pie_percent=[]
        self.pie_centroid = self.clst.cluster_centers_
        self.labels = list(self.clst.labels_)
        for i in range(len(self.pie_centroid)):
            j=self.labels.count(i)
            j=j/(len(self.labels))
            self.pie_percent.append(j)
        plt.pie(self.pie_percent,colors=np.array(self.pie_centroid/255), labels=np.arange(len(self.pie_centroid)))

    def transform(self):
        self.labels = self.clst.labels_.reshape((self.img_data.shape[0], self.img_data.shape[1]))
        centers = np.asarray(self.clst.cluster_centers_, dtype='int32')
        self.new_img_data = []
        for i in range(self.labels.shape[0]):
            row = []
            for j in range(self.labels.shape[1]):
                row.append(centers[self.labels[i][j]])
            self.new_img_data.append(row)
        self.new_img_data = np.asarray(self.new_img_data, dtype='int32')
        
        
        self.new_img = Image.fromarray(np.uint8(self.new_img_data)).convert('RGB')

        for _ in range(self.n_noise_filter):
            self.new_img = self.new_img.filter(ImageFilter.ModeFilter(size=self.filter_size))
        self.new_img_data = np.asarray(self.new_img, dtype='uint8')            
        return self.new_img_data

    def transform_bw(self):
        bounds_h = np.zeros(shape=(self.new_img_data.shape[0], self.new_img_data.shape[1]))
        bounds_v = np.zeros(shape=bounds_h.shape)
        i = 0
        for i in range(bounds_h.shape[0]):
            j = 0
            while j < (bounds_h.shape[1]-1):
                if np.sum(self.new_img_data[i, j, :] - self.new_img_data[i, j+1, :]) != 0:
                    bounds_h[i, j+1] = 1
                    j += 2
                else:
                    j += 1

        i = 0
        for i in range(bounds_h.shape[1]):
            j = 0
            while j < (bounds_h.shape[0] - 1):
                if  np.sum(self.new_img_data[j, i, :] - self.new_img_data[j+1, i, :]) != 0:
                    bounds_h[j+1, i] = 1
                    j += 2
                else:
                    j += 1

        self.bw_img_data = []
        for i in range(bounds_h.shape[0]):
            row = []
            for j in range(bounds_h.shape[1]):
                if bounds_h[i, j] == 1 or bounds_v[i, j] == 1:
                    row.append([0, 0, 0])
                else:
                    row.append([255, 255, 255])
            self.bw_img_data.append(row)
        
        self.bw_img_data = np.uint8(self.bw_img_data)
        return self.bw_img_data

    def add_labels(self):
        centroids = self.segmant_labeling()
        self.labeled_img = Image.fromarray(np.uint8(self.bw_img_data)).convert('RGB')
        draw_text = ImageDraw.Draw(self.labeled_img)
        for color in range(self.n_colors):
            color_centroids = centroids[color]
            for centroid in color_centroids:
                draw_text.text(
                    (int(centroid[1]), int(centroid[0])),
                    str(color),
                    fill=('#1C0606')
                    )

        self.labeled_img_data = np.asarray(self.labeled_img, dtype='uint8')
        return self.labeled_img_data

    
    def segmant_labeling(self):
        
        all_centroids = []
        for color in range(self.n_colors):
            im = np.where(self.labels == color, 0, 1)
            label_im, num = scipy.ndimage.label(im)
            centroids = scipy.ndimage.center_of_mass(im, label_im, range(1,num+1))
            centroids = list(centroids)
            all_centroids.append(centroids)        
        return all_centroids


    def load_image(self, filename):
        self.img = Image.open(filename).convert('RGB')
        width, height = self.img.size
        new_width, new_height = int(width*self.scale), int(height*self.scale)
        self.img = self.img.resize((new_width, new_height))
        self.img = self.img.filter(self.blurrer)
        self.img.load()
        self.img_data = np.asarray(self.img, dtype='int32')
        return self.img_data

    def plot_colors_chart(self):
        plt.pie(self.pie_percent,colors=np.array(self.pie_centroid/255), labels=np.arange(len(self.pie_centroid)))
        plt.show()


if __name__ == '__main__':

    filename = 'temp.png'


    clrfr = colorifer(n_colors=8, scale=1)

    clrfr.fit(filename)
    new_im_data = clrfr.transform()
    new_im = Image.fromarray(np.uint8(new_im_data)).convert('RGB')
    bw_data = clrfr.transform_bw()
    bw_im = Image.fromarray(np.uint8(bw_data)).convert('RGB')
    labeled_data = clrfr.add_labels()
    labeled_img = Image.fromarray(np.uint8(labeled_data)).convert('RGB')
    clrfr.plot_colors_chart()
    new_im.save('img1.jpg')
    bw_im.save('img2.jpg')
    labeled_img.save('img3.jpg')

