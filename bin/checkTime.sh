#!/bin/bash 

logFile=$(ls logs/*.log | sort | tail -n1)

awk_routine(){
    awk '
        BEGIN {
            FS=" ";
            printf("%40s ", "function name");
                printf("%10s ", "count");
                printf("%7s ", "total");
                printf("%10s ", "average");
                printf("\n");
        }
        NR != 1 { 
            functionName[$1]++;
            timeValue[$1] = timeValue[$1] + $2;
        }
        END {
            maxAve = 0;
            for (i in functionName){
                if(maxAve < timeValue[i]/functionName[i])
                    maxAve = timeValue[i]/functionName[i];
            }
            for (i in functionName){

                if (i == "") continue;

                N = 30;
                
                average = timeValue[i]/functionName[i];
                scaled  = int(N * average / maxAve);

                printf("%40s "  ,  i);
                printf("%10d "  , functionName[i]);
                printf("%7.2f " , timeValue[i]);
                printf("%10.2e [", average);
                for (j = 0; j < N; j++)
                    if (j < scaled)
                        printf("▥"); 
                    else
                        printf(" ");
                printf("]\n");
            }
        }
    '
}

cat $logFile | grep seconds | awk -F 'function' '{print $2}' | awk -F ' ' '{print $1, $3}' | awk_routine
