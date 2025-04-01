#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>
using namespace std;
class HopfieldNetwork {
private:
    int size;                // 网络维度（假设为正方形输入）
    int neuronCount;         // 神经元总数 = size * size
    int* neurons;            // 神经元状态数组（-1或1）
    double** weights;        // 权重矩阵（下三角存储）
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
    void train();                    //train 读
    void train(int n);               //train 读
    void recognize();                //in    读
    void recognize(int n);           //in    读
    void GetWeights();               //           out   输
    void GetPower();                 //in    读   out   输
    void GetPower(int n);            //in    读   out   输
    void GenerateData();             //           train 输
    void GenerateData(int n);        //           train 输
};
void HopfieldNetwork::train()
{
    freopen("train.txt", "r", stdin);
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
    delete[] pattern;
    fclose(stdin);
}
void HopfieldNetwork::train(int n)
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
void HopfieldNetwork::recognize()
{
    freopen("in.txt", "r", stdin);
    FILE* fp = freopen("out.txt", "w", stdout);
    char ch;
    const int maxIter = 100;                //maxIter 最大递归次数限制
    for (int i = 0; i < neuronCount; ++i)
    {
        cin >> ch;
        if (ch == '1')
            neurons[i] = 1;
        else
            neurons[i] = -1;
    }
    srand(time(0));
    for (int iter = 0; iter < maxIter; ++iter)
    {
        int updated = 0;
        int* order = new int[neuronCount];
        for (int i = 0; i < neuronCount; ++i) order[i] = i;
        for (int i = 0; i < neuronCount; ++i)
        {
            int j = rand() % neuronCount;
            swap(order[i], order[j]);        //随机异步更新
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
    fclose(stdin);
    fclose(fp);
}
void HopfieldNetwork::recognize(int n)
{
    freopen("in.txt", "r", stdin);
    FILE* fp = freopen("out.txt", "w", stdout);
    const int maxIter = 100;       //maxIter 最大递归次数限制
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
                swap(order[i], order[j]);        //随机异步更新
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
    for (int i = 0; i < neuronCount; ++i)
    {
        for (int j = 0; j <= i; ++j)
            cout << weights[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}
void HopfieldNetwork::GetPower()
{
    freopen("in.txt", "r", stdin);
    FILE* fp = freopen("out.txt", "w", stdout);
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
    cout << "能量为: " << -0.5 * power << endl;
    delete[] pattern;
    fclose(stdin);
    fclose(fp);
}
void HopfieldNetwork::GetPower(int n)
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
        cout << "能量为: " << -0.5 * power << endl;
        delete[] pattern;
        --n;
    }
    fclose(stdin);
    fclose(fp);
}
void HopfieldNetwork::GenerateData()
{
    FILE* fp = freopen("train.txt", "w", stdout);
    srand(time(0));
    for (int j = 0; j < size; ++j)
    {
        for (int i = 0; i < size; ++i)
        {
            int num = rand() % 3;
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
    fclose(fp);
}
void HopfieldNetwork::GenerateData(int n)
{
    FILE* fp = freopen("train.txt", "w", stdout);
    while (n)
    {
        mt19937 engine(random_device{}());
        uniform_int_distribution<int> distribution(0,1);
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