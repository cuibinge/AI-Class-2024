#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <random>
#include <Eigen/SVD>
extern "C" void calculatePseudoInverse(
    const int* input,   // 输入矩阵（一维数组）
    int rows,           // 矩阵行数
    int cols,           // 矩阵列数
    double* output      // 输出伪逆矩阵（预分配内存）
) {
    // 将输入数组转换为Eigen矩阵（动态大小，double类型）
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = static_cast<double>(input[i * cols + j]);
        }
    }
    // 计算奇异值分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd sigma = svd.singularValues().asDiagonal();
    // 计算伪逆矩阵
    Eigen::MatrixXd sigmaInv = Eigen::MatrixXd::Zero(cols, rows);
    double epsilon = 1e-10;  // 阈值用于判断奇异值是否为零
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        if (svd.singularValues()(i) > epsilon) {
            sigmaInv(i, i) = 1.0 / svd.singularValues()(i);
        }
    }
    Eigen::MatrixXd pseudoInv = svd.matrixV() * sigmaInv * svd.matrixU().adjoint();
    // 将结果复制到输出数组
    for (int i = 0; i < pseudoInv.rows(); ++i) {
        for (int j = 0; j < pseudoInv.cols(); ++j) {
            output[i * pseudoInv.cols() + j] = pseudoInv(i, j);
        }
    }
}
class MP_HopfieldNetwork {
private:
    int size;                // 网络维度（假设为正方形输入）
    int neuronCount;         // 神经元总数 = size * size
    int* neurons;            // 神经元状态数组（-1或1）
    double** weights;        // 权重矩阵（下三角存储）
public:
    MP_HopfieldNetwork(int imgSize) : size(imgSize)
    {
        neuronCount = size * size;
        neurons = new int[neuronCount];
        weights = new double* [neuronCount];
        for (int i = 0; i < neuronCount; ++i)
        {
            weights[i] = new double[i + 1]();
        }
    }
    ~MP_HopfieldNetwork()
    {
        delete[] neurons;
        for (int i = 0; i < neuronCount; ++i)
        {
            delete[] weights[i];
        }
        delete[] weights;
    }
    void train(int n);               //train     读
    void recognize(int n);           //in        读
    void GetWeights();               //              out   输
    void GetPower(int n);            //in        读  out   输
    void GenerateData(int n);        //              train 输
    void compare(int n);             //in/train  读  out   输
};
void MP_HopfieldNetwork::train(int n = 1) 
{
    freopen("train.txt", "r", stdin);
    int* pattern = new int[neuronCount * n];
    double* MP_pattern = new double[neuronCount * n];
    double** MP_weights = new double* [neuronCount];
    for (int i = 0; i < neuronCount; ++i)
    {
        MP_weights[i] = new double[neuronCount];
    }
    char ch;
    for (int i1 = 0; i1 < n; ++i1)
    {
        for (int i2 = 0; i2 < neuronCount; ++i2)
        {
            std::cin >> ch;
            if (ch == '1')
                pattern[i1 + i2 * n] = 1;
            else
                pattern[i1 + i2 * n] = -1;
        }
    }
    calculatePseudoInverse(pattern, neuronCount, n, MP_pattern);
    for (int i = 0; i < neuronCount; i++)
    {
        for (int j = 0; j < neuronCount; j++)
        {
            MP_weights[i][j] = 0.0;
            for (int k = 0; k < n; k++)
            {
                MP_weights[i][j] += pattern[i * n + k] * MP_pattern[k * neuronCount + j];
            }
        }
    }
    for (int i = 0; i < neuronCount; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            weights[i][j] = MP_weights[i][j];
        }
    }

    //FILE* fp = freopen("out.txt", "w", stdout);
    //////
    //for (int i = 0; i < neuronCount; ++i)
    //{
    //    for (int j = 0; j < n; ++j)
    //    {
    //        std::cout << pattern[i * n + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;
    //////
    //for (int i = 0; i < n; ++i)
    //{
    //    for (int j = 0; j < neuronCount; ++j)
    //    {
    //        std::cout << MP_pattern[i * neuronCount + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;
    //////
    //for (int i = 0; i < neuronCount; ++i)
    //{
    //    for (int j = 0; j < neuronCount; ++j)
    //    {
    //        std::cout << MP_weights[i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;
    //////
    //for (int i = 0; i < neuronCount; ++i)
    //{
    //    for (int j = 0; j <= i; ++j)
    //    {
    //        std::cout << weights[i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;
    //////
    //fclose(fp);

    delete[] pattern;
    delete[] MP_pattern;
    for (int i = 0; i < neuronCount; ++i)
    {
        delete[] MP_weights[i];
    }
    delete[] MP_weights;
    fclose(stdin);
}
void MP_HopfieldNetwork::recognize(int n = 1)
{
    freopen("in.txt", "r", stdin);
    FILE* fp = freopen("out.txt", "w", stdout);
    const int maxIter = 100;       //maxIter 最大递归次数限制
    while (n)
    {
        char ch;
        for (int i = 0; i < neuronCount; ++i)
        {
            std::cin >> ch;
            if (ch == '1')
                neurons[i] = 1;
            else
                neurons[i] = -1;
        }
        std::mt19937 engine(std::random_device{}());
        std::uniform_int_distribution<int> distribution(0, neuronCount - 1);
        for (int iter = 0; iter < maxIter; ++iter)
        {
            int updated = 0;
            int* order = new int[neuronCount];
            for (int i = 0; i < neuronCount; ++i) order[i] = i;
            for (int i = 0; i < neuronCount; ++i)
            {
                int j = distribution(engine);
                std::swap(order[i], order[j]);        //随机异步更新
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
                std::cout << (neurons[y * size + x] > 0 ? '1' : '-');
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        --n;
    }
    fclose(stdin);
    fclose(fp);
}
void MP_HopfieldNetwork::GetWeights()
{
    FILE* fp = freopen("out.txt", "w", stdout);
    for (int i = 0; i < neuronCount; ++i)
    {
        for (int j = 0; j <= i; ++j)
            std::cout << weights[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
    fclose(fp);
}
void MP_HopfieldNetwork::GetPower(int n = 1)
{
    freopen("in.txt", "r", stdin);
    FILE* fp = freopen("out.txt", "w", stdout);
    while (n)
    {
        int* pattern = new int[neuronCount];
        char ch;
        for (int i = 0; i < neuronCount; ++i)
        {
            std::cin >> ch;
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
        std::cout << "能量为: " << -0.5 * power << std::endl;
        delete[] pattern;
        --n;
    }
    fclose(stdin);
    fclose(fp);
}
void MP_HopfieldNetwork::GenerateData(int n = 1)
{
    FILE* fp = freopen("train.txt", "w", stdout);
    while (n)
    {
        std::mt19937 engine(std::random_device{}());
        std::uniform_int_distribution<int> distribution(0, 1);
        for (int j = 0; j < size; ++j)
        {
            for (int i = 0; i < size; ++i)
            {
                int num = distribution(engine);
                if (num == 1)
                {
                    std::cout << "1";
                }
                else
                    std::cout << "-";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        --n;
    }
    fclose(fp);
}
void MP_HopfieldNetwork::compare(int n = 1)
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
        cout << "第" << i + 1 << "个数据的相似度为: " << similarity << "%" << endl;
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