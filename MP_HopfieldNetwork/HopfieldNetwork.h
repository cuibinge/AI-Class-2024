#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <random>
using namespace std;
class HopfieldNetwork {
private:
    int size;                // ����ά�ȣ�����Ϊ���������룩
    int neuronCount;         // ��Ԫ���� = size * size
    int* neurons;            // ��Ԫ״̬���飨-1��1��
    double** weights;        // Ȩ�ؾ��������Ǵ洢��
public:
    HopfieldNetwork(int imgSize) : size(imgSize)
    {
        neuronCount = size * size;
        neurons = new int[neuronCount];
        weights = new double* [neuronCount];
        for (int i = 0; i < neuronCount; ++i)
        {
            weights[i] = new double[i + 1]();
        }
    }
    ~HopfieldNetwork()
    {
        delete[] neurons;
        for (int i = 0; i < neuronCount; ++i)
        {
            delete[] weights[i];
        }
        delete[] weights;
    }
    void train(int n);               //train     ��
    void recognize(int n);           //in        ��
    void GetWeights();               //              out   ��
    void GetPower(int n);            //in        ��  out   ��
    void GenerateData(int n);        //              train ��
    void compare(int n);             //in/train  ��  out   ��
};                                 
void HopfieldNetwork::train(int n = 1)
{
    freopen("train.txt", "r", stdin);
    int N = n;
    int* pattern = new int[neuronCount];
    while (N)
    {
        char ch;
        for (int i = 0; i < neuronCount; ++i)
        {
            cin >> ch;
            if (ch == '1')
                pattern[i] = 1;
            else
                pattern[i] = -1;
        }
        for (int i = 0; i < neuronCount; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                if (i != j)
                {
                    weights[i][j] += (pattern[i] * pattern[j]);
                }
            }
        }
        --N;
    }
    delete[] pattern;
    for (int i = 0; i < neuronCount; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            weights[i][j] /= neuronCount;
        }
    }
    fclose(stdin);
}
void HopfieldNetwork::recognize(int n = 1)
{
    freopen("in.txt", "r", stdin);
    FILE* fp = freopen("out.txt", "w", stdout);
    const int maxIter = 100;       //maxIter ���ݹ��������
    while (n)
    {
        char ch;
        for (int i = 0; i < neuronCount; ++i)
        {
            cin >> ch;
            if (ch == '1')
                neurons[i] = 1;
            else
                neurons[i] = -1;
        }
        //srand(time(0));
        mt19937 engine(random_device{}());
        uniform_int_distribution<int> distribution(0, neuronCount - 1);
        for (int iter = 0; iter < maxIter; ++iter)
        {
            int updated = 0;
            int* order = new int[neuronCount];
            for (int i = 0; i < neuronCount; ++i) order[i] = i;
            for (int i = 0; i < neuronCount; ++i)
            {
                int j = distribution(engine);
                //int j = rand() % neuronCount;
                swap(order[i], order[j]);        //����첽����
            }
            for (int n = 0; n < neuronCount; ++n)
            {
                int i = order[n];
                double sum = 0.0;
                for (int j = 0; j < neuronCount; ++j)
                {
                    if (j < i) sum += weights[i][j] * neurons[j];
                    else if (j > i) sum += weights[j][i] * neurons[j];
                }
                int newState = (sum >= 0) ? 1 : -1;
                if (newState != neurons[i])
                {
                    neurons[i] = newState;
                    updated++;
                }
            }
            delete[] order;
            if (updated == 0)
                break;
        }
        for (int y = 0; y < size; ++y)
        {
            for (int x = 0; x < size; ++x)
            {
                cout << (neurons[y * size + x] > 0 ? '1' : '-');
            }
            cout << endl;
        }
        cout << endl;
        --n;
    }
    fclose(stdin);
    fclose(fp);
}
void HopfieldNetwork::GetWeights()
{
    FILE* fp = freopen("out.txt", "w", stdout);
    for (int i = 0; i < neuronCount; ++i)
    {
        for (int j = 0; j <= i; ++j)
            cout << weights[i][j] << " ";
        cout << endl;
    }
    cout << endl;
    fclose(fp);
}
void HopfieldNetwork::GetPower(int n = 1)
{
    freopen("in.txt", "r", stdin);
    FILE* fp = freopen("out.txt", "w", stdout);
    while (n)
    {
        int* pattern = new int[neuronCount];
        char ch;
        for (int i = 0; i < neuronCount; ++i)
        {
            cin >> ch;
            if (ch == '1')
                pattern[i] = 1;
            else
                pattern[i] = -1;
        }
        double power = 0;
        for (int i = 0; i < neuronCount; ++i)
        {
            for (int j = 0; j < neuronCount; ++j)
            {
                if (i > j)
                    power += pattern[i] * pattern[j] * weights[i][j];
                else if (j > i)
                    power += pattern[i] * pattern[j] * weights[j][i];
            }
        }
        cout << "����Ϊ: " << -0.5 * power << endl;
        delete[] pattern;
        --n;
    }
    fclose(stdin);
    fclose(fp);
}
void HopfieldNetwork::GenerateData(int n = 1)
{
    FILE* fp = freopen("train.txt", "w", stdout);
    while (n)
    {
        mt19937 engine(random_device{}());
        uniform_int_distribution<int> distribution(0, 1);
        for (int j = 0; j < size; ++j)
        {
            for (int i = 0; i < size; ++i)
            {
                int num = distribution(engine);
                if (num == 1)
                {
                    cout << "1";
                }
                else
                    cout << "-";
            }
            cout << endl;
        }
        cout << endl;
        --n;
    }
    fclose(fp);
}
void HopfieldNetwork::compare(int n = 1)
{
    FILE* fp1 = freopen("out.txt", "w", stdout);
    ////
    freopen("train.txt", "r", stdin);
    int** t_neuron = new int* [n];
    for (int i = 0; i < n; ++i)
    {
        t_neuron[i] = new int[neuronCount];
    }
    char ch;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < neuronCount; ++j)
        {
            cin >> ch;
            t_neuron[i][j] = ((ch == '1') ? 1 : -1);
        }
    }
    fclose(stdin);
    ////
    freopen("in.txt", "r", stdin);
    int** c_neuron = new int* [n];
    for (int i = 0; i < n; ++i)
    {
        c_neuron[i] = new int[neuronCount];
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < neuronCount; ++j)
        {
            cin >> ch;
            c_neuron[i][j] = ((ch == '1') ? 1 : -1);
        }
    }
    fclose(stdin);
    ////
    for (int i = 0; i < n; ++i)
    {
        double similarity = 0;
        int same = 0;
        for (int j = 0; j < neuronCount; ++j)
        {
            same += ((t_neuron[i][j] == c_neuron[i][j]) ? 1 : 0);
        }
        similarity = 100.0 * same / neuronCount;
        cout << "��" << i + 1 << "�����ݵ����ƶ�Ϊ: " << similarity << "%" << endl;
    }
    for (int i = 0; i < n; ++i)
    {
        delete[] t_neuron[i];
    }
    delete[] t_neuron;
    for (int i = 0; i < n; ++i)
    {
        delete[] c_neuron[i];
    }
    delete[] c_neuron;
    fclose(fp1);
}