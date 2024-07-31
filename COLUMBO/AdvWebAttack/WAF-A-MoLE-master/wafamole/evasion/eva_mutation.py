"""The main class of WAF-A-MoLE"""
import signal
import time

from multiprocessing import Pool

# from wafamole.evasion.engine import CoreEngine
# from wafamole.models import Model
import sys
sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/WAF-A-MoLE-master')
# sys.path.append('/home/ustc-5/XiaoF/AdvWebDefen/AdvSqli/mutate/')
# # from zw_detector import ZWDetector
# from ml_detector import MLDetector

from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer
from wafamole.utils.check import type_check

from abc import ABCMeta, abstractmethod
# from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer
# from wafamole.models import Model
# map = Pool().map

class CoreEngineNew(object, metaclass=ABCMeta):

    def __init__(self, model):
        self._model = model

    def _mutation_round(self, payload, round_size):
        fuzzer = SqlFuzzer(payload)
        # print(payload)
        # Some mutations do not apply to some payloads
        # This removes duplicate payloads
        payloads = {fuzzer.fuzz() for _ in range(round_size)}
        # print(payloads, len(payloads))
        # print(self._model.thre, self._model.ml_alg)
        # exit(0)
        # results = map(self._model.get_score, payloads)
        results = [self._model.get_score(x) for x in payloads]
        confidence, payload = min(zip(results, payloads))
        return confidence, payload

    @abstractmethod
    def evaluate(self, payload, max_rounds, round_size, timeout, threshold):
        """It tries to produce a payloads that should be classified as a benign payload.

        Arguments:
            payload (str) : the initial payload
            max_rounds (int) : maximum number of mutation rounds
            round_size (int) : how many mutation for each round
            timeout (int) : number of seconds before the timeout
            threshold (float) : default 0.5, customizable for different results

        Returns:
            float, str : minimum confidence and correspondent payload that achieve that score
        """
        raise NotImplementedError("")

class RandomEvasionEngineNew(CoreEngineNew):
    def __init__(self, model):
        self._transformations = []
        super(RandomEvasionEngineNew, self).__init__(model)

    def evaluate(self, payload, max_rounds, round_size, timeout, threshold):
        self._transformations = []

        current_time = time.time()
        print('Start round', time.time())
        while(time.time() < current_time + timeout):
        # for _ in range(max_rounds):
            # print(time.time(), current_time, current_time + timeout)
            conf, payload = self._mutation_round(payload, 1)
            self._transformations.append((conf, payload, time.time()))
        print('End round', time.time())
        return min(self._transformations)

    @property
    def transformations(self):
        return self._transformations

class EvasionEngineNew(CoreEngineNew):
    """Evasion engine object.
    """

    def __init__(self, model):
        """Initialize an evasion object.
        Arguments:
            model: the input model to evaluate

        Raises:
            TypeError: model is not Model
        """
        super(EvasionEngineNew, self).__init__(model)

    # def _mutation_round(self, payload, round_size):
    #
    #     fuzzer = SqlFuzzer(payload)
    #
    #     # Some mutations do not apply to some payloads
    #     # This removes duplicate payloads
    #     payloads = {fuzzer.fuzz() for _ in range(round_size)}
    #     results = map(self.model.classify, payloads)
    #     confidence, payload = min(zip(results, payloads))
    #
    #     return confidence, payload

    def evaluate(
        self,
        payload: str,
        max_rounds: int = 1000,
        round_size: int = 20,
        timeout: int = 14400,
        threshold: float = 0.5,
    ):
        """It tries to produce a payloads that should be classified as a benign payload.

        Arguments:
            payload (str) : the initial payload
            max_rounds (int) : maximum number of mutation rounds
            round_size (int) : how many mutation for each round
            timeout (int) : number of seconds before the timeout
            threshold (float) : default 0.5, customizable for different results

        Raises:
            TypeError : input arguments are mistyped.

        Returns:
            float, str : minimum confidence and correspondent payload that achieve that score
        """

        type_check(payload, str, "payload")
        type_check(max_rounds, int, "max_rounds")
        type_check(round_size, int, "round_size")
        type_check(timeout, int, "timeout")
        type_check(threshold, float, "threshold")

        def _signal_handler(signum, frame):
            raise TimeoutError()

        # Timeout setup
        signal.signal(signal.SIGALRM, _signal_handler)
        signal.alarm(timeout)

        init_score = self._model.get_score(payload)
        # benign
        if init_score <= threshold:
            run_res = {'success': False, 'except': None, 'benign': True}
            return run_res
        
        evaluation_results = []
        min_confidence, min_payload = self._mutation_round(payload, round_size)
        evaluation_results.append((min_confidence, min_payload))

        run_res = {'success': False, 'except': None, 'benign': False}
        try:
            while max_rounds > 0 and min_confidence > threshold:
                for candidate_confidence, candidate_payload in sorted(
                    evaluation_results
                ):
                    max_rounds -= 1

                    confidence, payload = self._mutation_round(
                        candidate_payload, round_size
                    )
                    if confidence < candidate_confidence:
                        evaluation_results.append((confidence, payload))
                        min_confidence, min_payload = min(evaluation_results)
                        break

            if min_confidence < threshold:
                print("[+] Threshold reached")
                run_res = {'success': True, 'except': None, 'benign': False, 
                           'min_score': min_confidence, 'min_payload': min_payload}
            elif max_rounds <= 0:
                except_str = "[!] Max number of iterations reached"
                print(except_str)
                run_res = {'success': False, 'except': None, 'benign': False, 
                           'min_score': min_confidence, 'min_payload': min_payload}

        except TimeoutError:
            except_str = "[!] Execution timed out"
            print(except_str)
            run_res = {'success': False, 'except': None, 'benign': False, 
                        'min_score': min_confidence, 'min_payload': min_payload}

        print(
            "Reached confidence {}\nwith payload\n{}".format(
                min_confidence, repr(min_payload)
            )
        )

        return run_res