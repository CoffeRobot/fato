def to_string(A, precision=3):

        I = J = 1
        tmp = A.shape
        mess = ''

        prec_str = '{:0.' + str(precision) + 'f}'

        if len(tmp) == 1:
            I = tmp[0]
            for i in range(0,I):
                mess += prec_str.format(A[i]) + ' '
        else:
            I = tmp[0]
            J = tmp[1]
            for i in range(0,I):
                for j in range(0,J):
                    mess += prec_str.format(A[i,j]) + ' '
                mess += '\n'

        return mess