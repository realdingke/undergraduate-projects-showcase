#data=['Alabama[edit]',
# 'Auburn (Auburn University)[1]',
# 'Florence (University of North Alabama)',
# 'Jacksonville (Jacksonville State University)[2]',
# 'Livingston (University of West Alabama)[2]',
# 'Montevallo (University of Montevallo)[2]',
# 'Troy (Troy University)[2]',
# 'Tuscaloosa (University of Alabama, Stillman College, Shelton State)[3][4]',
# 'Tuskegee (Tuskegee University)[5]',
# 'Alaska[edit]',
# 'Fairbanks (University of Alaska Fairbanks)[2]',
# 'Arizona[edit]',
# 'Flagstaff (Northern Arizona University)[6]',
# 'Tempe (Arizona State University)',
# 'Tucson (University of Arizona)',
# 'Arkansas[edit]',
# 'Arkadelphia (Henderson State University, Ouachita Baptist University)[2]',
# 'Conway (Central Baptist College, Hendrix College, University of Central Arkansas)[2]',
# 'Fayetteville (University of Arkansas)[7]',
# 'Jonesboro (Arkansas State University)[8]',
# 'Magnolia (Southern Arkansas University)[2]',
# 'Monticello (University of Arkansas at Monticello)[2]',
# 'Russellville (Arkansas Tech University)[2]',
# 'Searcy (Harding University)[5]',
# 'California[edit]',
# 'Angwin (Pacific Union College)[2]',
# 'Arcata (Humboldt State University)[5]',
# 'Berkeley (University of California, Berkeley)[5]',
# 'Chico (California State University, Chico)[2]',
# 'Claremont (Claremont McKenna College, Pomona College, Harvey Mudd College, Scripps College, Pitzer College, Keck Graduate Institute, Claremont Graduate University)[5]',
# 'Cotati (California State University, Sonoma)[2]',
# 'Davis (University of California, Davis)[1]',
# 'Irvine (University of California, Irvine)',
# 'Isla Vista (University of California, Santa Barbara)[2]',
# 'University Park, Los Angeles (University of Southern California)',
# 'Merced (University of California, Merced)',
# 'Orange (Chapman University)',
# 'Palo Alto (Stanford University)',
# 'Pomona (Cal Poly Pomona, WesternU)[9][10][11] and formerly Pomona College',
# 'Redlands (University of Redlands)',
# 'Riverside (University of California, Riverside, California Baptist University, La Sierra University)',
# 'Sacramento (California State University, Sacramento)',
# 'University District, San Bernardino (California State University, San Bernardino, American Sports University)',
# 'San Diego (University of California, San Diego, San Diego State University)',
# 'San Luis Obispo (California Polytechnic State University)[2]',
# 'Santa Barbara (Fielding Graduate University, Santa Barbara City College, University of California, Santa Barbara, Westmont College)[2]',
# 'Santa Cruz (University of California, Santa Cruz)[2]',
# 'Turlock (California State University, Stanislaus)',
# 'Westwood, Los Angeles (University of California, Los Angeles)[2]',
# 'Whittier (Whittier CollegeRio Hondo College)',
# 'Colorado[edit]',
# 'Alamosa (Adams State College)[2]',
# 'Boulder (University of Colorado at Boulder)[12]',
# 'Durango (Fort Lewis College)[2]',
# 'Fort Collins (Colorado State University)[13]',
# 'Golden (Colorado School of Mines)',
# 'Grand Junction (Colorado Mesa University)',
# 'Greeley (University of Northern Colorado)',
# 'Gunnison (Western State College)[2]',
# 'Pueblo, Colorado (Colorado State University-Pueblo)']
#dict1={}
#for line in data:
#    if '[edit]' in line:
#        state=line[:line.index('[')]
#    dict1[line]=state
#print(dict1)
def month_to_quarter(data):
    month=data[data.index('-')+1:]
    if month=='01' or month=='02' or month=='03':
        return data[:data.index('-')]+'q1'
    if month=='04' or month=='05' or month=='06':
        return data[:data.index('-')]+'q2'
    if month=='07' or month=='08' or month=='09':
        return data[:data.index('-')]+'q3'
    if month=='10' or month=='11' or month=='12':
        return data[:data.index('-')]+'q4'
    return data*2

print(month_to_quarter('1999-06'))
