"""
Week 1 practice project template for Python Data Visualization
Load a county-level PNG map of the USA and draw it using matplotlib
"""

import matplotlib.pyplot as plt

# Houston location

USA_SVG_SIZE = [555, 352]
HOUSTON_POS = [302, 280]



def draw_USA_map(map_name):
    """
    Given the name of a PNG map of the USA (specified as a string),
    draw this map using matplotlib
    """
    # Load map image, note that using 'rb'option in open() is critical since png files are binary
    with open(map_name,'rb') as pngfile:
        img=plt.imread(pngfile)

    #  Get dimensions of USA map image
        dimensions=img.shape             #dimensions is an iterable with (height-y,width-x,channels)
    # Plot USA map
        plt.imshow(img)
    # Plot green scatter point in center of map
        xc=(dimensions[1])/2
        yc=(dimensions[0])/2
        plt.scatter(xc,yc,marker='*',c='g')
     
    # Plot red scatter point on Houston, Tx - include code that rescale coordinates for larger PNG files
        xh=(HOUSTON_POS[0]/USA_SVG_SIZE[0])*dimensions[1]
        yh=(HOUSTON_POS[1]/USA_SVG_SIZE[1])*dimensions[0]
        plt.scatter(xh,yh,marker='*',c='r')
        
        plt.show()
    pass

draw_USA_map("USA_Counties_555x352.png")
draw_USA_map("USA_Counties_1000x634.png")   

