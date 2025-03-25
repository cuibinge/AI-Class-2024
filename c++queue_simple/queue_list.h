#include <iostream>
#include <cstdio>
using namespace std;
template<typename T>
class node
{
public:
	T _value;
	node <T>* _next;
	node() { _next = nullptr; }
};
template<typename T>
class myqueue2
{
public:
	node <T>* _head;
	node <T>* _tail;
	myqueue2()
	{
		_head = nullptr;
		_tail = nullptr;
	}
	~myqueue2()
	{
		node <T>* ptr1 = _head;
		while (ptr1 != nullptr)
		{
			node <T>* ptr2 = ptr1->_next;
			delete ptr1;
			ptr1 = ptr2;
		}
	}
	void push(T value);              //��Ӷ�βԪ��(���)
	void pop();                      //�Ƴ�����Ԫ��(����)
	T& front() const;                 //���ʶ���Ԫ��
	T& back() const;                  //���ʶ�βԪ��
	bool empty() const;              //�ж϶����Ƿ�Ϊ��
	int size() const;                //��ȡ����Ԫ�ظ���
};
template<typename T>
void myqueue2<T>::push(T value)
{
	if (_head == nullptr)
	{
		_head = new node<T>;
		_tail = _head;
		_tail->_value = value;
	}
	else
	{
		_tail->_next = new node<T>;
		_tail = _tail->_next;
		_tail->_value = value;
	}
}
template<typename T>
void myqueue2<T>::pop()
{
	if (_head == nullptr)
		return;
	node <T>* ptr = _head;
	_head = _head->_next;
	delete ptr;
	if (_head == nullptr)
		_tail = nullptr;
}
template<typename T>
T& myqueue2<T>::front() const
{
	if (empty())
		throw runtime_error("Myqueue is empty!\n");
	return _head->_value;
}
template<typename T>
T& myqueue2<T>::back() const
{
	if (empty())
		throw runtime_error("Myqueue is empty!\n");
	return _tail->_value;
}
template<typename T>
bool myqueue2<T>::empty() const
{
	if (_head == nullptr)
		return true;
	else
		return false;
}
template<typename T>
int myqueue2<T>::size() const
{
	int i = 0;
	for (node <T>* ptr = _head; ptr != nullptr; ptr = ptr->_next, i++) {}
	return i;
}