fib=[0,1];
count=0;
for num in fib:
    if count<20:
        total=fib[-1]+fib[-2];
        fib.append(total)
    else:
        pass    
    count=count+1;
    pass
print(fib)