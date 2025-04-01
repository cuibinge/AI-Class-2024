
#include "HopfieldNetwork.h"
int main()
{
    HopfieldNetwork net(14);
    //net.GenerateData(12);
    net.train(12);  
    //net.GetPower(2);
    net.recognize(1);
    return 0;
}