lowercase = 'abcdef'
digits = '01234'
#lowercaselist=[x+y for x in lowercase for y in lowercase]
#digitslist=[str(x)+str(y) for x in digits for y in digits]
#list0=[chars+nums for chars in lowercaselist for nums in digitslist]
#list1=[a+b+c+d for a in lowercase for b in lowercase for c in digits for d in digits]
#list0=[]
#for char in lowercase:
#    for num in digits:
#        list0.append(char+num)
#list1=[char+num for char in lowercase for num in digits]
list0=[]
for char1 in lowercase:
    for char2 in lowercase:
        for num1 in digits:
            for num2 in digits:
                list0.append(char1+char2+num1+num2)
print(list0)
#print(list0==list1)