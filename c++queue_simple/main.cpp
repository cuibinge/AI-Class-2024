#include "queue_list.h"        //myqueue2
#include "queue_deque.h"       //myqueue1
int main()
{
	myqueue2 <double> A;
	for (int i = 1; i <= 800; i++)
	{A.push(1.0 * i / 2);}
	cout << A.front() << endl;
	return 0;
}