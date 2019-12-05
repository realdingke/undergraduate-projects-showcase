"""
Project for Week 3 of "Python Data Visualization".
Unify data via common country name.

Be sure to read the project description page for further information
about the expected behavior of the program.
"""
import csv
import math
import pygal

def reconcile_countries_by_name(plot_countries, gdp_countries):
    """
    Inputs:
      plot_countries - Dictionary whose keys are plot library country codes
                       and values are the corresponding country name
      gdp_countries  - Dictionary whose keys are country names used in GDP data

    Output:
      A tuple containing a dictionary and a set.  The dictionary maps
      country codes from plot_countries to country names from
      gdp_countries The set contains the country codes from
      plot_countries that were not found in gdp_countries.
    """
    dict0={}
    set0=set()
    for key,value in plot_countries.items():
        if value in gdp_countries:
            dict0[key]=value
        else:
            set0.add(key)
    print("Country codes that are missing from gdp data:",set0)
    return (dict0,set0)


def read_csv_as_nested_dict(filename,keyfield,separator,quote):
    """
    This function reads a csv file returns it as nested dictionary
    """
    with open(filename,'r',newline='') as csvfile:
        csvread=csv.DictReader(csvfile,delimiter=separator,quotechar=quote)
        nested_dict={}
        for row in csvread:
            nested_dict[row[keyfield]]=row
    return nested_dict


def build_map_dict_by_name(gdpinfo, plot_countries, year):
    """
    Inputs:
      gdpinfo        - A GDP information dictionary
      plot_countries - Dictionary whose keys are plot library country codes
                       and values are the corresponding country name
      year           - String year to create GDP mapping for

    Output:
      A tuple containing a dictionary and two sets.  The dictionary
      maps country codes from plot_countries to the log (base 10) of
      the GDP value for that country in the specified year.  The first
      set contains the country codes from plot_countries that were not
      found in the GDP data file.  The second set contains the country
      codes from plot_countries that were found in the GDP data file, but
      have no GDP data for the specified year.
    """
    nested_dict=read_csv_as_nested_dict(gdpinfo["gdpfile"],gdpinfo["country_name"],
                                        gdpinfo["separator"],gdpinfo["quote"])
    return_dict={}
    set0=set()
    set1=set()
    for key,name in plot_countries.items():
        if name in nested_dict:
            if (nested_dict[name])[year]!='':
                value=math.log10(float((nested_dict[name])[year]))
                return_dict[key]=value
            else:
                set1.add(key)        
        else:
            set0.add(key)
        
    return (return_dict,set0,set1)


def render_world_map(gdpinfo, plot_countries, year, map_file):
    """
    Inputs:
      gdpinfo        - A GDP information dictionary
      plot_countries - Dictionary whose keys are plot library country codes
                       and values are the corresponding country name
      year           - String year to create GDP mapping for
      map_file       - Name of output file to create

    Output:
      Returns None.

    Action:
      Creates a world map plot of the GDP data for the given year and
      writes it to a file named by map_file.
    """
    tuple0=build_map_dict_by_name(gdpinfo,plot_countries,year)
    mapchart=pygal.maps.world.World()
    mapchart.title="Countries' GDP plot on the world map in the year "+year+"unified by common country name."
    mapchart.add("Countries with GDP data in "+year, tuple0[0])
    mapchart.add("Countries missing from GDP data", tuple0[1])
    mapchart.add("Countries with no GDP data for "+year, tuple0[2])
    mapchart.render_to_file(map_file)
    mapchart.render_in_browser()
    pass


def test_render_world_map():
    """
    Test the project code for several years.
    """
    gdpinfo = {
        "gdpfile": "isp_gdp.csv",
        "separator": ",",
        "quote": '"',
        "min_year": 1960,
        "max_year": 2015,
        "country_name": "Country Name",
        "country_code": "Country Code"
    }

    # Get pygal country code map
    pygal_countries = pygal.maps.world.COUNTRIES

    # 1960
    render_world_map(gdpinfo, pygal_countries, "1960", "isp_gdp_world_name_1960.svg")

    # 1980
    render_world_map(gdpinfo, pygal_countries, "1980", "isp_gdp_world_name_1980.svg")

    # 2000
    render_world_map(gdpinfo, pygal_countries, "2000", "isp_gdp_world_name_2000.svg")

    # 2010
    render_world_map(gdpinfo, pygal_countries, "2010", "isp_gdp_world_name_2010.svg")


# Make sure the following call to test_render_world_map is commented
# out when submitting to OwlTest/CourseraTest.

test_render_world_map()
