TAB='    '
blocks = ['aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb']

def imports():
    print(
        'from rspn.uhf_ccsd.equations.cc_jacobian.doubles_doubles import ('  #)
    )
    for left in blocks:
        for right in blocks:
            print(f'{TAB}get_cc_j_doubles_doubles_{left}{right},')
    print(')')


def calls():
    for left in blocks:
        for right in blocks:
            print(f'''
        {left}_{right} = get_cc_j_doubles_doubles_{left}{right}(
            self.uhf_scf_data,
            self.uhf_ccsd_data,
        ).reshape(dims['{left}'], dims['{right}'])''')


def jacobian():
    dd_block = ''
    for left in blocks:
        for right in blocks:
            dd_block += f' {left}_{right},'
        dd_block += '\n'
    dd_block = dd_block[:-1]
    print(dd_block)

if __name__ == "__main__":
    jacobian()
