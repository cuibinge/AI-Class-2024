#define _CRT_SECURE_NO_WARNINGS
#include "HopfieldNetwork.h"
#include "Moore-Penrose.h"
int main()
{
    MP_HopfieldNetwork net(14);         //伪逆法
    //HopfieldNetwork net(14);            //传统法

    int n = 10;
    //net.GenerateData(n);                //已经生成了 196 个数据
    net.train(n); 
    net.recognize(n);
    //net.GetWeights();
    //net.compare(n);
    return 0;
}