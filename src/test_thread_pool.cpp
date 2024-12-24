#include <iostream>
#include <chrono>
#include "thread_pool.h"

void exampleTask(int n) {
    std::this_thread::sleep_for(std::chrono::seconds(n));
    std::cout << "Task completed after " << n << " seconds" << std::endl;
}


void run(ThreadPool & pool) {
    std::vector<std::future<void>> results;

    for (int i = 1; i <= 8; ++i) {
        results.emplace_back(pool.enqueue(exampleTask, i));
    }
    std::this_thread::sleep_for(std::chrono::seconds(4));
}

int main() {
    ThreadPool pool(8); // 创建一个包含4个线程的线程池

    run(pool);
    //std::this_thread::sleep_for(std::chrono::seconds(10));
    /*
    for (auto &&result : results) {
        result.get(); // 等待所有任务完成
    }
    */
    std::cout << "Task end " << std::endl;
    return 0;
}