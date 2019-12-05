"""
Project for Week 4 of "Python Programming Essentials".
Collection of functions to process dates.

Be sure to read the project description page for further information
about the expected behavior of the program.
"""

import datetime

def days_in_month(year_input,month_input):
    """
    Inputs:
      year  - an integer between datetime.MINYEAR and datetime.MAXYEAR
              representing the year
      month - an integer between 1 and 12 representing the month

    Returns:
      The number of days in the input month.
    """
    if month_input==12:
        date_1=datetime.date(year_input,month_input,1);
        date_2=datetime.date((year_input+1),1,1);
    else:
        date_1=datetime.date(year_input,month_input,1);
        date_2=datetime.date(year_input,(month_input+1),1);
    diff=(date_2-date_1);
    number_of_days=diff.days;
    return number_of_days

print(days_in_month(2019,12))


def is_valid_date(year, month, day):
    """
    Inputs:
      year  - an integer representing the year
      month - an integer representing the month
      day   - an integer representing the day

    Returns:
      True if year-month-day is a valid date and
      False otherwise
    """
    if (datetime.MINYEAR<=year)and(year<=datetime.MAXYEAR):
        boolean=True
    if 1<=month<=12:
        boolean=True
    if (datetime.MINYEAR<=year<=datetime.MAXYEAR)and(1<=month<=12):
        if 1<=day<=days_in_month(year,month):
            boolean=True
        else:
            boolean=False
    else:
        boolean=False
    return boolean
print(is_valid_date(1999,6,8))

def days_between(year1, month1, day1, year2, month2, day2):
    """
    Inputs:
      year1  - an integer representing the year of the first date
      month1 - an integer representing the month of the first date
      day1   - an integer representing the day of the first date
      year2  - an integer representing the year of the second date
      month2 - an integer representing the month of the second date
      day2   - an integer representing the day of the second date

    Returns:
      The number of days from the first date to the second date.
      Returns 0 if either date is invalid or the second date is
      before the first date.
    """
    return 0

def age_in_days(year, month, day):
    """
    Inputs:
      year  - an integer representing the birthday year
      month - an integer representing the birthday month
      day   - an integer representing the birthday day

    Returns:
      The age of a person with the input birthday as of today.
      Returns 0 if the input date is invalid or if the input
      date is in the future.
    """
    return 0
