from enum import IntEnum

quad_length = dict(
    # mechanical length: Q4
    Q4=0.5,
    # mechanical length: Q1
    Q1=0.25,
    # mechanical length: Q2
    Q2=0.2,
    # mechanical length: Q3
    Q3=0.25,
    # mechanical length: Q5
    Q5=0.2,
)

class Polarity(IntEnum):
    pos = 1
    neg = -1

quad_polarities = dict(
    # horizontal focusing
    Q1=Polarity.pos,
    # vertical focusing
    Q2=Polarity.neg,
    # vertical focusing
    Q3=Polarity.neg,
    # horizontal focusing
    Q4=Polarity.pos,
    # vertical focusing
    Q5=Polarity.neg,
)

def get_polarity_by_magnet_name(name):
    return quad_polarities[name[:2].upper()]

def get_length_by_magnet_name(name):
    return quad_length[name[:2].upper()]
