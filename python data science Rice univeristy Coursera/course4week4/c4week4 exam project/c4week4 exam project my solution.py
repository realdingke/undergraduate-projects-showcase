"""
Project for Week 4 of "Python Data Visualization".
Unify data via common country codes.

Be sure to read the project description page for further information
about the expected behavior of the program.
"""

import csv
import math
import pygal

def read_csv_as_nested_dict(filename,separator,quote,keyfield="name"):
    """
    This function reads a csv file returns it as nested dictionary
    """
    with open(filename,'r',newline='') as csvfile:
        csvread=csv.DictReader(csvfile,delimiter=separator,quotechar=quote)
        nested_dict={}
        for row in csvread:
            nested_dict[row[keyfield]]=row
    return nested_dict


def build_country_code_converter(codeinfo):
    """
    Inputs:
      codeinfo      - A country code information dictionary

    Output:
      A dictionary whose keys are plot country codes and values
      are world bank country codes, where the code fields in the
      code file are specified in codeinfo.
    """
    nested_dict0=read_csv_as_nested_dict(filename=codeinfo["codefile"],
                                         separator=codeinfo["separator"],
                                         quote=codeinfo["quote"],keyfield=codeinfo["plot_codes"])
    return_dict={}
    for key,value in nested_dict0.items():
        return_dict[value[codeinfo["plot_codes"]]]=value[codeinfo["data_codes"]]
        
    return return_dict


def reconcile_countries_by_code(codeinfo, plot_countries, gdp_countries):
    """
    Inputs:
      codeinfo       - A country code information dictionary
      plot_countries - Dictionary whose keys are plot library country codes
                       and values are the corresponding country name
      gdp_countries  - Dictionary whose keys are country codes used in GDP data

    Output:
      A tuple containing a dictionary and a set.  The dictionary maps
      country codes from plot_countries to country codes from
      gdp_countries.  The set contains the country codes from
      plot_countries that did not have a country with a corresponding
      code in gdp_countries.

      Note that all codes should be compared in a case-insensitive
      way.  However, the returned dictionary and set should include
      the codes with the exact same case as they have in
      plot_countries and gdp_countries.
    """
    return_dict={}
    return_set=set()
    code_dict=build_country_code_converter(codeinfo)
    code_dict_upper={key.upper():value.upper() for key,value in code_dict.items()}
    for key0 in plot_countries:
        checker=False
        for key1 in gdp_countries:
            if code_dict_upper[key0.upper()]==key1.upper():
                return_dict[key0]=key1
                checker=True
        if not checker:
            return_set.add(key0)
                
    return (return_dict,return_set)


def build_map_dict_by_code(gdpinfo, codeinfo, plot_countries, year):
    """
    Inputs:
      gdpinfo        - A GDP information dictionary
      codeinfo       - A country code information dictionary
      plot_countries - Dictionary mapping plot library country codes to country names
      year           - String year for which to create GDP mapping

    Output:
      A tuple containing a dictionary and two sets.  The dictionary
      maps country codes from plot_countries to the log (base 10) of
      the GDP value for that country in the specified year.  The first
      set contains the country codes from plot_countries that were not
      found in the GDP data file.  The second set contains the country
      codes from plot_countries that were found in the GDP data file, but
      have no GDP data for the specified year.
    """
    gdp_nested_dict=read_csv_as_nested_dict(filename=gdpinfo["gdpfile"],separator=gdpinfo["separator"],
                                            quote=gdpinfo["quote"],keyfield=gdpinfo["country_code"])
    tuple0=reconcile_countries_by_code(codeinfo, plot_countries, gdp_nested_dict)
    set0=tuple0[1]
    dict0=tuple0[0]
    return_dict={}
    set1=set()
    for key,value in dict0.items():
        gdp_year=(gdp_nested_dict[value])[year]
        if gdp_year!='':
            return_dict[key]=math.log(float(gdp_year),10)
        else:
            set1.add(key)
    return (return_dict, set0, set1)



def render_world_map(gdpinfo, codeinfo, plot_countries, year, map_file):
    """
    Inputs:
      gdpinfo        - A GDP information dictionary
      codeinfo       - A country code information dictionary
      plot_countries - Dictionary mapping plot library country codes to country names
      year           - String year of data
      map_file       - String that is the output map file name

    Output:
      Returns None.

    Action:
      Creates a world map plot of the GDP data in gdp_mapping and outputs
      it to a file named by svg_filename.
    """
    tuple0=build_map_dict_by_code(gdpinfo,codeinfo,plot_countries,year)
    mapchart=pygal.maps.world.World()
    mapchart.title="Countries' GDP plot(log10 scale) on the world map in the year "+year+", unified by common country codes."
    mapchart.add("Countries with GDP data in "+year, tuple0[0])
    mapchart.add("Countries missing from GDP data", tuple0[1])
    mapchart.add("Countries with no GDP data in "+year, tuple0[2])
    mapchart.render_to_file(map_file)
    mapchart.render_in_browser()
    return


def test_render_world_map():
    """
    Test the project code for several years
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

    codeinfo = {
        "codefile": "isp_country_codes.csv",
        "separator": ",",
        "quote": '"',
        "plot_codes": "ISO3166-1-Alpha-2",
        "data_codes": "ISO3166-1-Alpha-3"
    }

    # Get pygal country code map
    pygal_countries = pygal.maps.world.COUNTRIES

    # 1960
    render_world_map(gdpinfo, codeinfo, pygal_countries, "1960", "isp_gdp_world_code_1960.svg")

    # 1980
    render_world_map(gdpinfo, codeinfo, pygal_countries, "1980", "isp_gdp_world_code_1980.svg")

    # 2000
    render_world_map(gdpinfo, codeinfo, pygal_countries, "2000", "isp_gdp_world_code_2000.svg")

    # 2010
    render_world_map(gdpinfo, codeinfo, pygal_countries, "2010", "isp_gdp_world_code_2010.svg")


# Make sure the following call to test_render_world_map is commented
# out when submitting to OwlTest/CourseraTest.

test_render_world_map()
