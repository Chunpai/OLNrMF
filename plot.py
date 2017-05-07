import matplotlib.pyplot as plt

def plotNetwork(data_dir,M,anomaly,n,l):
    plt.axis([0,n+1,0,l+1])
    plt.xlabel('user')
    plt.ylabel('movie')
    count = 0
    for source in M:
        source_list = []
        target_list = []
        for target in M[source]:
            count += 1
            source_list.append(source)
            target_list.append(target)
        plt.plot(source_list, target_list, 'b.', markersize=0.5 )
    print 'total edges in M',count
    for source in anomaly:
        source_list = []
        target_list = []
        for target in anomaly[source]:
            source_list.append(source)
            target_list.append(target)
        plt.plot(source_list, target_list, 'r.', markersize=1.0)
    plt.savefig(data_dir+'injected_M.png')

 
def plotResidual(data_dir,R,anomaly,n,r,l):
    plt.axis([0,n+1,0,l+1])
    count = 0
    acount = 0
    #outfile = open(data_dir+'residuals2/R'+str(r)+'.csv','w')
    #outfile2 = open(data_dir+'result2.csv','a')
    for source in R:
        source_list = []
        target_list = []
        anomaly_source_list = []
        anomaly_target_list = []
        for target in R[source]:
            if R[source][target] > 0.00001:
                count += 1
                source_list.append(source)
                target_list.append(target)
                outfile.write(str(source)+';'+str(target)+'\n')
                if source in anomaly:
                    if target in anomaly[source]:
                        acount += 1
                        anomaly_source_list.append(source)
                        anomaly_target_list.append(target)
        plt.plot(source_list, target_list, 'b.', markersize=0.5)
        plt.plot(anomaly_source_list, anomaly_target_list, 'r.', markersize=0.5)
    outfile.close()
    #plt.savefig(data_dir+'residuals2/R'+str(r)+'.png')
    plt.show()
    #outfile2.write(str(r)+';'+str(count)+';'+str(acount)+'\n')
    #outfile2.close()
    print r
    print count
    print acount

