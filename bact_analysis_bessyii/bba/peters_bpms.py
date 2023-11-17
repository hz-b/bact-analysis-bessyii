pattern = re.compile("\s*(\d+)\s*(\S+)\s*(\S+)")
txt =  """1             2.736        BPMZ5D1R
 2             3.995        BPMZ6D1R
 3             6.474        BPMZ7D1R
 4             7.675        BPMZ1T1R
 5             8.591        BPMZ2T1R
 6             11.005       BPMZ3T1R
 7             12.639       BPMZ4T1R
 8             17.361       BPMZ5T1R
 9             18.995       BPMZ6T1R
 10            21.474       BPMZ7T1R
 11            22.675       BPMZ1D2R
 12            23.591       BPMZ2D2R
 13            26.005       BPMZ3D2R
 14            27.264       BPMZ4D2R
 15            32.736       BPMZ5D2R
 16            33.995       BPMZ6D2R
 17            36.474       BPMZ7D2R
 18            37.675       BPMZ1T2R
 19            38.591       BPMZ2T2R
 20            41.005       BPMZ3T2R
 21            42.639       BPMZ4T2R
 22            47.361       BPMZ5T2R
 23            48.995       BPMZ6T2R
 24            51.474       BPMZ7T2R
 25            52.675       BPMZ1D3R
 26            53.591       BPMZ2D3R
 27            56.005       BPMZ3D3R
 28            57.264       BPMZ4D3R
 29            62.736       BPMZ5D3R
 30            63.995       BPMZ6D3R
 31            66.474       BPMZ7D3R
 32            67.675       BPMZ1T3R
 33            68.591       BPMZ2T3R
 34            71.005       BPMZ3T3R
 35            72.639       BPMZ4T3R
 36            74.1275      BPMZ41T3R
 37            75.3795      BPMZ42T3R
 38            77.361       BPMZ5T3R
 39            78.995       BPMZ6T3R
 40            81.474       BPMZ7T3R
 41            82.675       BPMZ1D4R
 42            83.591       BPMZ2D4R
 43            86.005       BPMZ3D4R
 44            87.264       BPMZ4D4R
 45            92.736       BPMZ5D4R
 46            93.995       BPMZ6D4R
 47            96.474       BPMZ7D4R
 48            97.675       BPMZ1T4R
 49            98.591       BPMZ2T4R
 50            101.005      BPMZ3T4R
 51            102.639      BPMZ4T4R
 52            107.361      BPMZ5T4R
 53            108.995      BPMZ6T4R
 54            111.474      BPMZ7T4R
 55            112.675      BPMZ1D5R
 56            113.591      BPMZ2D5R
 57            116.005      BPMZ3D5R
 58            117.693      BPMZ4D5R
 59            122.307      BPMZ5D5R
 60            123.995      BPMZ6D5R
 61            126.474      BPMZ7D5R
 62            127.675      BPMZ1T5R
 63            128.591      BPMZ2T5R
 64            131.005      BPMZ3T5R
 65            132.639      BPMZ4T5R
 66            137.361      BPMZ5T5R
 67            138.995      BPMZ6T5R
 68            141.474      BPMZ7T5R
 69            142.675      BPMZ1D6R
 70            143.591      BPMZ2D6R
 71            146.005      BPMZ3D6R
 72            147.264      BPMZ4D6R
 73            152.736      BPMZ5D6R
 74            153.995      BPMZ6D6R
 75            156.474      BPMZ7D6R
 76            157.675      BPMZ1T6R
 77            158.591      BPMZ2T6R
 78            161.005      BPMZ3T6R
 79            162.639      BPMZ4T6R
 80            167.361      BPMZ5T6R
 81            168.995      BPMZ6T6R
 82            171.474      BPMZ7T6R
 83            172.675      BPMZ1D7R
 84            173.591      BPMZ2D7R
 85            176.005      BPMZ3D7R
 86            177.264      BPMZ4D7R
 87            182.736      BPMZ5D7R
 88            183.995      BPMZ6D7R
 89            186.474      BPMZ7D7R
 90            187.675      BPMZ1T7R
 91            188.591      BPMZ2T7R
 92            191.005      BPMZ3T7R
 93            192.639      BPMZ4T7R
 94            197.361      BPMZ5T7R
 95            198.995      BPMZ6T7R
 96            201.474      BPMZ7T7R
 97            202.675      BPMZ1D8R
 98            203.591      BPMZ2D8R
 99            206.005      BPMZ3D8R
 100           207.264      BPMZ4D8R
 101           212.736      BPMZ5D8R
 102           213.995      BPMZ6D8R
 103           216.474      BPMZ7D8R
 104           217.675      BPMZ1T8R
 105           218.591      BPMZ2T8R
 106           221.005      BPMZ3T8R
 107           222.639      BPMZ4T8R
 108           227.361      BPMZ5T8R
 109           228.995      BPMZ6T8R
 110           231.474      BPMZ7T8R
 111           232.675      BPMZ1D1R
 112           233.591      BPMZ2D1R
 113           236.005      BPMZ3D1R
 114           237.264      BPMZ4D1R"""

def convert_peter_line(line):
    idx, s, name = pattern.match(line).groups()
    return int(idx), float(s), name
peter_bpms = (
    [convert_peter_line(line)[2] for line in txt.split("\n")]

)
peter_bpms