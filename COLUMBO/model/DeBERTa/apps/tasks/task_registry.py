import importlib
import os
import sys
from glob import glob

from .task import Task
import sys
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/GenerAT-main/')
from DeBERTa.utils import get_logger

__all__ = ['load_tasks', 'load_tasks_new', 'register_task', 'get_task']
tasks = {}

logger = get_logger()


def register_task(name=None, desc=None):
    def register_task_x(cls):
        global tasks
        _name = name
        if _name is None:
            _name = cls.__name__

        _desc = desc
        if _desc is None:
            _desc = _name

        _name = _name.lower()
        if _name in tasks:
            logger.warning(f'{_name} already registered in the registry: {tasks[_name]}.')
        assert issubclass(cls, Task), f'Registered class must be a subclass of Task.'
        tasks[_name] = cls
        cls._meta = {
            'name': _name,
            'desc': _desc}
        # print('register: ', tasks)
        return cls

    if type(name) == type:
        cls = name
        name = None
        return register_task_x(cls)
    return register_task_x


def load_tasks(task_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys_tasks = glob(os.path.join(script_dir, "*.py"))
    for t in sys_tasks:
        m = os.path.splitext(os.path.basename(t))[0]
        if not m.startswith('_'):
            importlib.import_module(f'DeBERTa.apps.tasks.{m}')
            # print('load: ', m, '\t', tasks, '\n')

    if task_dir:
        assert os.path.exists(task_dir), f"{task_dir} must be a valid directory."
        customer_tasks = glob(os.path.join(task_dir, "*.py"))
        sys.path.append(task_dir)
        for t in customer_tasks:
            m = os.path.splitext(os.path.basename(t))[0]
            if not m.startswith('_'):
                importlib.import_module(f'{m}')

def load_tasks_new(task_dir=None): # importlib.import_module(f'apps.tasks.{m}') for advsqli
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys_tasks = glob(os.path.join(script_dir, "*.py"))
    for t in sys_tasks:
        m = os.path.splitext(os.path.basename(t))[0]
        if not m.startswith('_'):
            importlib.import_module(f'apps.tasks.{m}')
            # print('load: ', m, '\t', tasks, '\n')

    if task_dir:
        assert os.path.exists(task_dir), f"{task_dir} must be a valid directory."
        customer_tasks = glob(os.path.join(task_dir, "*.py"))
        sys.path.append(task_dir)
        for t in customer_tasks:
            m = os.path.splitext(os.path.basename(t))[0]
            if not m.startswith('_'):
                importlib.import_module(f'{m}')


def get_task(name=None):
    if name is None:
        return tasks
    return tasks[name.lower()]
