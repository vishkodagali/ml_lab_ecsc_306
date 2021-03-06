data_1 = [(0.000000,95.364693) , 
    (1.000000,99.217205) , 
    (.000000,65.195834), 
    (3.000000,50.105519) , 
    (4.000000,39.342380), 
    (5.000000,27.400286), 
    (6.000000,41.057128), 
    (7.000000,15.500619), 
    (8.000000,4.259608), 
    (9.000000,0.639151), 
    (10.000000,-8.409936), 
    (11.000000, -3.383926), 
    (12.000000,-12.8555597), 
    (13.000000,-27.758333), 
    (14.000000,-55.606221)] 
 
data_2 = [(2104.,400.), 
     (1600.,330.), 
     (2400.,369.), 
     (1416.,232.), 
     (3000.,540.)] 
 
def create_hypothesis(t1, t0): 
    return lambda x: t1*x + t0 
 
def linregression(data, l_rate=0.001, variance=0.00001): 
 
 
    t0_guess = 1. 
    t1_guess = 1. 
 
 
    t0_last = 100. 
    t1_last = 100. 
 
    m = len(data) 
 
    while (abs(t1_guess-t1_last) > variance or abs(t0_guess - t0_last) > variance): 
 
        t1_last = t1_guess 
        t0_last = t0_guess 
 
        hypothesis = create_hypothesis(t1_guess, t0_guess) 
 
        t0_guess = t0_guess - l_rate * (1./m) * sum([hypothesis(point[0]) - point[1] for point in data]) 
        t1_guess = t1_guess - l_rate * (1./m) * sum([ (hypothesis(point[0]) - point[1]) * point[0] for point in data])    
 
    return ( t0_guess,t1_guess ) 
 
 
 
points = [(float(x),float(y)) for (x,y) in data_1] 
 
res = linregression(points) 
print res