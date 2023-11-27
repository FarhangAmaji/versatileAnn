class VAnnTsDataloader(DataLoader):
    ...
    def _testTimeWithANumWorker(self, nw, num_iterations):
        # kwargs={key: getattr(self, key) for key in ['batch_size', 'collate_fn', 'sampler','createBatchStructEverytime']}
        # if isinstance(kwargs['collate_fn'],types.MethodType) and \
        #         kwargs['collate_fn'].__self__.__class__ == VAnnTsDataloader and \
        #         kwargs['collate_fn'].__func__.__name__ == 'commonCollate_fn':
        #     del kwargs['collate_fn']
        # new_dl = VAnnTsDataloader(self.dataset, num_workers=nw, **kwargs)

        self.num_workers = nw
        start_time = time.perf_counter()

        iteration_count = 0
        while iteration_count < num_iterations:
            for batch in self:
                iteration_count += 1
                print(iteration_count, batch)
                if iteration_count >= num_iterations:
                    break

        end_time = time.perf_counter()

        return end_time - start_time


    def bestNumWorkerFinder(self, num_iterations=20):
        import multiprocessing
        numCores = multiprocessing.cpu_count()
        results = {}

        for num_workers in list(range(2, numCores * 2 + 1, 2)):
            training_time = self._testTimeWithANumWorker(num_workers, num_iterations)
            results[num_workers] = training_time

        # Find the best num_workers value
        bestNum_workers = min(results, key=results.get)
        self.num_workers = bestNum_workers
        print('best_num_workers set', bestNum_workers)