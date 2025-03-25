#include <iostream>
#include <cstdio>
using namespace std;
#define size_block 512
template<typename T>
class myblock
{
public:
	T* _array;
	int _a_first;
	int _a_last;
	myblock()
	{
		_array = nullptr;
		_a_first = 0;
		_a_last = 0;
	}
	~myblock()
	{
		delete[] _array;
	}
};
template<typename T>
class myqueue1
{
public:
	myblock <T>** _map;
	int _m_num;
	const int size_myblock;
	myqueue1() :size_myblock(size_block / sizeof(T))
	{
		_map = nullptr;
		_m_num = 0;
	}
	~myqueue1()
	{
		for (int i = 1; i <= _m_num; i++)
			delete _map[i - 1];
		delete[] _map;
	}
	void push(T value);              //添加队尾元素(入队)
	void pop();                      //移除队首元素(出队)
	T& front() const;                 //访问队首元素
	T& back() const;                  //访问队尾元素
	bool empty() const;              //判断队列是否为空
	int size() const;                //获取队列元素个数
};
template<typename T>
void myqueue1<T>::push(T value)
{
	if (_map == nullptr)
	{
		_m_num += 1;
		_map = new myblock<T>*[_m_num];
		_map[_m_num - 1] = new myblock<T>;
		_map[_m_num - 1]->_array = new T[size_myblock];
		_map[_m_num - 1]->_a_last += 1;
		_map[_m_num - 1]->_a_first += 1;
		_map[_m_num - 1]->_array[_map[_m_num - 1]->_a_last - 1] = value;
	}
	else if (_map[_m_num - 1]->_a_last < size_myblock)
	{
		_map[_m_num - 1]->_a_last += 1;
		_map[_m_num - 1]->_array[_map[_m_num - 1]->_a_last - 1] = value;
	}
	else
	{
		_m_num += 1;
		myblock <T>** ptr = _map;
		_map = new myblock<T>*[_m_num];
		for (int i = 1; i <= _m_num - 1; i++)
			_map[i - 1] = ptr[i - 1];
		delete[] ptr;
		_map[_m_num - 1] = new myblock<T>;
		_map[_m_num - 1]->_array = new T[size_myblock];
		_map[_m_num - 1]->_a_last += 1;
		_map[_m_num - 1]->_a_first += 1;
		_map[_m_num - 1]->_array[_map[_m_num - 1]->_a_last - 1] = value;
	}
}
template<typename T>
void myqueue1<T>::pop()
{
	if (_map == nullptr)
		return;
	else if (_map[0]->_a_first == _map[0]->_a_last && _m_num == 1)
	{
		_m_num -= 1;
		delete _map[0];
		delete[] _map;
		_map = nullptr;
	}
	else if (_map[0]->_a_first == _map[0]->_a_last)
	{
		_m_num -= 1;
		myblock <T>** ptr = _map;
		_map = new myblock<T>*[_m_num];
		for (int i = 1; i <= _m_num; i++)
			_map[i - 1] = ptr[i];
		for (int i = 1; i <= _m_num + 1; i++)
			delete ptr[i - 1];
		delete ptr[0];
	}
	else
		_map[0]->_a_first += 1;
}
template<typename T>
T& myqueue1<T>::front() const
{
	if (empty())
		throw runtime_error("Myqueue is empty!\n");
	return _map[0]->_array[(_map[0]->_a_first) - 1];
}
template<typename T>
T& myqueue1<T>::back() const
{
	if (empty())
		throw runtime_error("Myqueue is empty!\n");
	return _map[_m_num - 1]->_array[(_map[_m_num - 1]->_a_last) - 1];
}
template<typename T>
bool myqueue1<T>::empty() const
{
	if (_map == nullptr)
		return true;
	else
		return false;
}
template<typename T>
int myqueue1<T>::size() const
{
	if (_m_num == 0)
		return 0;
	else if (_m_num == 1)
		return _map[0]->_a_last - _map[0]->_a_first + 1;
	else
		return
		size_myblock * (_m_num - 2) +
		_map[0]->_a_last - _map[0]->_a_first +
		_map[_m_num - 1]->_a_last - _map[_m_num - 1]->_a_first + 2;
}