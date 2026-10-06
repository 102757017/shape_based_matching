/*
 * 极简线程池(本仓库新增)。
 *
 * 为什么不用 OpenMP:
 *   MSVC 的 OpenMP 在默认等待策略下, 并行区结束后工作线程会原地自旋若干毫秒。
 *   椭圆检测每帧只有两个并行区、总时长几十毫秒, 实测自旋开销能吃掉全部并行收益
 *   ——必须设环境变量 OMP_WAIT_POLICY=passive 才有效, 而第三方调用方不该被要求
 *   设置环境变量。此外用 OpenMP 会让发布产物依赖 vcomp140.dll。
 *
 *   这里的替代方案: 线程在并行区之间睡在 condition_variable 上(不自旋), 线程在
 *   进程内复用, 用动态取块做负载均衡, 且只依赖标准库。
 *
 * 约定: parallelFor 只保证每个下标被恰好调用一次; 任务之间不得写同一块内存
 *      (调用方负责把结果先落到各自的槽位, 再按序合并, 见 detect.cpp)。
 */
#ifndef _INCLUDE_PARALLEL_H_
#define _INCLUDE_PARALLEL_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace zgh {

class ThreadPool {
 public:
  // 进程内单例。故意堆分配且从不析构: 避免静态对象析构顺序导致仍在运行的工作
  // 线程访问已销毁的成员。
  static ThreadPool& instance() {
    static ThreadPool* p = new ThreadPool();
    return *p;
  }

  int hardwareThreads() const { return hw_; }

  // total 个下标交给 requested_threads 个线程处理。<=0 表示用满硬件并发数。
  void parallelFor(int total, const std::function<void(int)>& body,
                   int requested_threads = 0) {
    if (total <= 0) {
      return;
    }
    int nw = requested_threads > 0 ? requested_threads : hw_;
    if (nw < 1) {
      nw = 1;
    }
    if (nw > total) {
      nw = total;           // 起比任务数还多的线程没有意义
    }
    if (nw <= 1) {
      for (int i = 0; i < total; ++i) {
        body(i);
      }
      return;
    }
    ensureWorkers(nw);

    // 分发锁: 允许调用方自己多线程并发调用(比如同时检测多张图), 这些并行区会
    // 排队执行而不是把 body_/next_ 踩乱。注意不要在 body 里再发起 parallelFor
    // (嵌套) —— 会死锁。
    std::unique_lock<std::mutex> dispatch(dispatch_mu_);
    std::unique_lock<std::mutex> lk(mu_);
    body_ = &body;
    total_ = total;
    active_ = nw;
    finished_ = 0;
    // 细任务(如梯度逐行)按块领取, 减少 atomic 争抢
    chunk_ = total > 4096 ? 64 : (total > 256 ? 16 : 1);
    next_.store(0);
    ++generation_;
    active_generation_ = generation_;
    cv_start_.notify_all();
    cv_done_.wait(lk, [this]() { return finished_ >= active_; });
    body_ = nullptr;
    active_generation_ = 0;
  }

  // 把 num_threads=0/负数 解析成实际使用的线程数
  static int resolveThreads(int requested) {
    if (requested > 0) {
      return requested;
    }
    const int hw = instance().hardwareThreads();
    return hw > 0 ? hw : 1;
  }

 private:
  ThreadPool() : hw_(static_cast<int>(std::thread::hardware_concurrency())) {
    if (hw_ < 1) {
      hw_ = 1;
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void ensureWorkers(int n) {
    std::unique_lock<std::mutex> lk(mu_);
    while (worker_count_ < n) {
      std::thread(&ThreadPool::workerLoop, this, worker_count_).detach();
      ++worker_count_;
    }
  }

  void workerLoop(int id) {
    unsigned seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_start_.wait(lk, [&]() {
        // 三个条件缺一不可:
        //  1. 有新一代派发
        //  2. 这一代还没结束(否则晚启动的线程会读到已清空的 body_)
        //  3. 本线程的编号在本轮需要的线程数内 —— cv_start_ 是 notify_all,
        //     之前更大规模派发留下的空闲线程也会被唤醒。若不加这条限制, 它们
        //     也会 ++finished_, 使 completed 计数被提前满足, 主线程会在工作
        //     还没做完时就返回(表现为随机崩溃 / 结果错乱)。
        return generation_ != seen && active_generation_ == generation_ &&
               id < active_;
      });
      seen = generation_;
      const std::function<void(int)>* body = body_;
      const int total = total_;
      const int chunk = chunk_;
      lk.unlock();

      // 动态领取任务块, 天然负载均衡(各候选的工作量差异很大)
      for (;;) {
        const int begin = next_.fetch_add(chunk);
        if (begin >= total) {
          break;
        }
        const int end = begin + chunk < total ? begin + chunk : total;
        for (int i = begin; i < end; ++i) {
          (*body)(i);
        }
      }

      lk.lock();
      ++finished_;
      if (finished_ >= active_) {
        cv_done_.notify_one();
      }
    }
  }

  int hw_ = 1;
  std::mutex dispatch_mu_;
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  int worker_count_ = 0;

  const std::function<void(int)>* body_ = nullptr;
  std::atomic<int> next_{0};
  int total_ = 0;
  int chunk_ = 1;
  int active_ = 0;
  int finished_ = 0;
  unsigned generation_ = 0;
  unsigned active_generation_ = 0;
};

}  // namespace zgh

#endif  // _INCLUDE_PARALLEL_H_
