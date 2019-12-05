"""
Week 1 practice project template for Python Data Visualization
Load a county-level PNG map of the USA and draw it using matplotlib
Week 4 practice project
draw cancer risk data for each county on USA map
"""

import matplotlib
import matplotlib.pyplot as plt
import csv
import math

# Houston location

USA_SVG_SIZE = [555, 352]
HOUSTON_POS = [302, 280]
SCATTER_POINT_SIZE=0.25
SCATTER_POPULATION=5000
MAX_CANCER_RISK=0.00015
MIN_CANCER_RISK=0.0000086

def read_csv_as_nested_list(filename,separator,quote):
    """
    This function reads a csvfile as nested list
    """
    with open(filename,'r',newline='') as csvfile:
        csvread=csv.reader(csvfile,delimiter=separator,quotechar=quote)
        nested_list=[]
        for row in csvread:
            nested_list.append(row)
    return nested_list

def compute_county_cirle(county_population):
    """
    This function takes the population of one county,
    and returns the corresponding area of scatter point on map
    """
    scatter_size=((SCATTER_POINT_SIZE)/SCATTER_POPULATION)*county_population
    return scatter_size

def create_riskmap(colormap):
    """
    This function takes a colormap
    returns the function to map the cancer risk data to RGB value on the colormap
    """
    norm=matplotlib.colors.Normalize(vmin=math.log(MIN_CANCER_RISK,10),vmax=math.log(MAX_CANCER_RISK,10))
    scalarmap=matplotlib.cm.ScalarMappable(norm=norm,cmap=colormap)
    return (lambda cancer_risk_log: scalarmap.to_rgba(cancer_risk_log))

def compute_cvalue(cancer_risk_data):
    """
    This function takes cancer risk data as float
    returns the final value for c option in scatter plot
    """
    color_map=matplotlib.cm.jet
    RGB_function=create_riskmap(color_map)
    return RGB_function(math.log(cancer_risk_data,10))


def draw_cancer_risk_map(joined_csv_file_name,map_name,num_counties=3140):
    """
    Given the name of a PNG map of the USA (specified as a string), also a cancer risk csv file
    draw this map using matplotlib
    plot scatter points at each county center

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
    # Read in the cancer risk table and plot scatter points of county center
        nested_list0=read_csv_as_nested_list(joined_csv_file_name,separator=',',quote='"')
        for row in nested_list0:
            if nested_list0.index(row)<num_counties:
                x=(float(row[5])/USA_SVG_SIZE[0])*dimensions[1]
                y=(float(row[6])/USA_SVG_SIZE[1])*dimensions[0]
                plt.scatter(x,y,c=compute_cvalue(float(row[4])),s=compute_county_cirle(int(row[3])))

        plt.show()
        #plt.savefig('cancer_risk_USA_map.pdf')
    pass


draw_cancer_risk_map("cancer_risk_joined.csv","USA_Counties_555x352.png")
draw_cancer_risk_map("cancer_risk_joined.csv","USA_Counties_1000x634.png",500)
