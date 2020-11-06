import re


def _exp_moving_avg_meter(existing_value: float, input_value: float, alpha: float) -> float:
    new_value = existing_value * alpha + input_value * (1 - alpha)
    return new_value




# --------------------------------------------------------------------------
def string_type_loss_split(loss_description: str) -> list:
    delete_space = re.sub(r'\s', '', loss_description)
    add_tmp_mark = re.sub(r'\+', r'^+', delete_space)
    add_tmp_mark = re.sub(r'-', r'^-', add_tmp_mark)
    return re.split(r'\^', add_tmp_mark)




# ---------------------------------------------------------------------------
class ExpMovingAvgMeter(object):
    """
    Exponential Moving Average Meter class
    new value = existing_value * alpha + input_value * (1-alpha)
    """
    def __init__(self, alpha=0.99):
        self.value = None
        self.alpha = alpha

    def update(self, val):
        if self.value is None:
            self.value = val
        else:
            self.value = _exp_moving_avg_meter(self.value, val, self.alpha)



