#define _CRT_SECURE_NO_WARNINGS
#include "HopfieldNetwork.h"
#include "Moore-Penrose.h"
int main()
{
    MP_HopfieldNetwork net(14);         //α�淨
    //HopfieldNetwork net(14);            //��ͳ��

    int n = 10;
    //net.GenerateData(n);                //�Ѿ������� 196 ������
    net.train(n); 
    net.recognize(n);
    //net.GetWeights();
    //net.compare(n);
    return 0;
}