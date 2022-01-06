import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage import data, color, io
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean

# Lädt das Bild und konvertiert es in ein schwarz-weiß Bild
filename = os.path.join('/Users/dustinwinkler/Desktop/Programmieren/Python/Hausarbeit/Bilder_Bibliothek', 'auto_04.jpg')
car = io.imread(filename)
car_grey = rgb2gray(car)
image = img_as_ubyte(car_grey)
edges = canny(image, sigma=3, low_threshold=0.7, high_threshold=0.8)

# Es wird ein Array erstellt und mit Werten zwischen 50px - 200px in 2er Schritten gefüllt
hough_radii = np.arange(50, 200, 2)
# hough_cricle ist eine mathematische Funktion, welche Kreisstrukturen im Wertebereich des Arrays hough_radii findet
# 'edges' gibt alle Kanten wieder, welche der Canny-Filter gefunden hat
hough_res = hough_circle(edges, hough_radii)

# Wählt 10 Konturen aus, welche die höchste Wahrscheinlichkeit eines Kreises aufweisen
# Rückgabe erfolgt für: die Wahrscheinlichkeit, dass es ein Kreis ist; x-Wert; y-Wert; Radius; - von den 10 wahrscheinlichsten Kreisen - 
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=10)

# Es werden die Kreise ignoriert, welche einen ähnlichen Mittelpunkt haben
# Es werden die x und y Koordinaten aller 10 Kreismittelpunkte eingelesen
x = np.array([cx, cy]).transpose()
# die 'distance'-Funktion erstellt eine 2D Matrix und erfasst die Entfernungen zwischen den einzelnen Mittelpunkten zueinander
distances = distance.pdist(x)
distance_matrix = distance.squareform(distances)
# circle_out ist eine leere Set-Liste 
circle_out = set()
# das 'i' dient als Vergleichsmittelpunkt, anschließend wird untersucht, welche Punkte innerhalb des Radius unserem Vergleichsobjekt (radii_temp) liegen
for i, radii_temp in enumerate(radii) :
    # im weiteren Verlauf werden alle nachfolgenden Mittelpunkte untersucht
    for j in range(i+1, len(radii)) :
        if distance_matrix [i, j] < radii_temp :
            # wenn die Distanz zwischen Vergleichsobjekt und nachfolgendem Mittelpunkt kleiner ist als der Radius des Vergleichsobjektes, wird er in die Liste hinzugefügt
            # circle_out enthält alle Kreise, welche entfernt werden sollen
            circle_out.add(j)

# ist die Anzahl der Radien, welche wir behalten
circle_count = len(radii) - len(circle_out)

# Zeichnet alle 10 Kreise
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(image)
# da 'enumerate' nur einen Wert akzeptiert, wird eine Liste an Werten mit dem 'zip'-Befehl erstellt
for i, kreis in enumerate(zip(cy, cx, radii)) :
    # für den aktuellen Kreis im Schleifendurchlauf wird der x- und y-Wert des Mittelpunktes sowie der Radius gespeichert
    center_y, center_x, radius = kreis
    # 'circy' und 'circx' speichern anschließend die Pixelwerte, welche für den Kreis eingefärbt werden müssen
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)

    # wenn 'i' nicht in der Liste der auszuschließenden Kreise ist, wird es rot gezeichnet, sonst schwarz
    if i not in circle_out :                                
        image[circy, circx] = (220, 20, 20)
    else :
        image[circy, circx] = (0,0,0)

# Ausgabe der Werte und Darstellung des Bildes
ax.set_title('Die Anzahl der Achsen beträgt: ' + str(circle_count))
ax.imshow(image)
plt.show()