#include <iostream>
#include <glog/logging.h>
using namespace std;
int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    int a=10;
    std::cout << "test"<< std::endl;
}