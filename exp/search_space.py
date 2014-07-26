#! /usr/bin/evn python
"""parameter space for cross validation"""
import re
import sys

#define the grid search item
#each grid item is a paramter with its parameter space
class search_item(object):
    __slots__ = ('name','start_val','step_val','end_val','val_num')

    def __init__(self, name, str_start_val, str_step_val, str_end_val):
        self.name = name
        self.start_val = float(str_start_val)
        self.step_val = float(str_step_val)
        if self.step_val == 1:
            raise ValueError('step value should not be 1')
        self.end_val = float(str_end_val)

        self.__calc_size()

    def val(self, index):
        ret = self.start_val
        while index > 0:
            ret *= self.step_val
            index -= 1
        return ret

    def size(self):
        return self.val_num

    def __calc_size(self):
        if self.end_val <= self.start_val:
            return 0
        else:
            self.val_num = 0 
            val = self.start_val
            while val <= self.end_val:
                val *= self.step_val
                self.val_num += 1

    def __str__(self):
        return 'name: {0} value: {1}:{2}:{3}'\
                .format(self.name,self.start_val, self.step_val,self.end_val)
    
class SearchSpace(object):
    #space dim: number of parameters to search
    #space size: number of grid items in the search space
    #search_space: search space
    __slots__ = ('space_dim','size','search_space')

    def __init__(self, param_space):
        self.__parse_search_space(param_space)

        self.space_dim = len(self.search_space)

        if self.space_dim > 0:
            self.size = reduce(lambda x, y: x * y, [item.size() for item in self.search_space])
        else:
            self.size = 0

    def get_param_cmd(self, grid_item_id):
        cmd  = ''
        for j in range(0,self.space_dim):
            dim_size = self.search_space[j].size()
            coor = grid_item_id % dim_size
            grid_item_id = int(grid_item_id / dim_size)

            cmd += '{0} {1} '.format(self.search_space[j].name,
                    self.search_space[j].val(coor))
        return cmd

    #parse the search space from input parameters
    def __parse_search_space(self,param_space):
        argv = filter(None,param_space.split(' '))
        #detect param
        param_pattern   = r'(?P<param_name>-\w+)'
        num_pattern     = r'\d*\.?\d+'
        search_pattern  = r'(?P<start_val>{0}):(?P<step_val>{1}):(?P<end_val>{2})'\
                .format(num_pattern,num_pattern,num_pattern)

        self.search_space = []
        k = 0 
        while k < len(argv):
            param   = re.match(param_pattern, argv[k])
            k += 1
            search  = re.match(search_pattern, argv[k])
            k += 1

            if param and search:
                item = search_item(param.group('param_name'),
                        search.group('start_val'),
                        search.group('step_val'),
                        search.group('end_val'))

                self.search_space.append(item)
            else:
                raise ValueError('incorrect input parameter {0} {1}'.format(argv[k-2],argv[k-1]))

if __name__ == '__main__':
    param_space = '-a 1:2:16 -b 0.5:2:10'
    ss = SearchSpace(param_space)
    for k in range(0,ss.size):
        cmd = ss.get_param_cmd(k)
        print cmd
