import itertools
TAB='    '
PAD=f'{TAB}'
oeblocks = ['aa', 'bb', ]
teblocks = ['aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb']

def imports():
    print(
        'from rspn.uhf_ccsd.equations.lHeecc.e1e1 import ('  # )
    )
    for left in oeblocks:
        for right in oeblocks:
            print(f'{TAB}get_lhe1e1cc_{left}{right},')
    print(')')
    print(
        'from rspn.uhf_ccsd.equations.lHeecc.e1e2 import ('  # )
    )
    for left in oeblocks:
        for right in teblocks:
            print(f'{TAB}get_lhe1e2cc_{left}{right},')
    print(')')
    print(
        'from rspn.uhf_ccsd.equations.lHeecc.e2e1 import ('  # )
    )
    for left in teblocks:
        for right in oeblocks:
            print(f'{TAB}get_lhe2e1cc_{left}{right},')
    print(')')
    print(
        'from rspn.uhf_ccsd.equations.lHeecc.e2e2 import ('  # )
    )
    for left in teblocks:
        for right in teblocks:
            print(f'{TAB}get_lhe2e2cc_{left}{right},')
    print(')')


def calls():
    for left, right in itertools.product(oeblocks, oeblocks):
        print(f'{TAB}f_{left}_{right} = get_lhe1e1cc_{left}{right}(**kwargs)')
    print()
    for left, right in itertools.product(oeblocks, teblocks):
        print(f'{TAB}f_{left}_{right} = get_lhe1e2cc_{left}{right}(**kwargs)')
    print()
    for left, right in itertools.product(teblocks, oeblocks):
        print(f'{TAB}f_{left}_{right} = get_lhe2e1cc_{left}{right}(**kwargs)')
    print()
    for left, right in itertools.product(teblocks, teblocks):
        print(f'{TAB}f_{left}_{right} = get_lhe2e2cc_{left}{right}(**kwargs)')
    print()


def einsum_template(sum: str, left: str, right: str, PAD: str) -> str:
    return f"""
{PAD}np.einsum(
{PAD}{TAB}'{sum}',
{PAD}{TAB}t_res_A[first]['{left}'],
{PAD}{TAB}f_{left}_{right},
{PAD}{TAB}t_res_B[second]['{right}'],
{PAD})
{PAD}+"""

def builder():
    sums = ''
    for left, right in itertools.product(oeblocks, oeblocks):
        sums += einsum_template('ai,aibj,bj->', left, right, TAB * 3)

    for left, right in itertools.product(oeblocks, teblocks):
        sums += einsum_template('ai,aibckj,bckj->', left, right, TAB * 3)

    for left, right in itertools.product(teblocks, oeblocks):
        sums += einsum_template('abji,abjick,ck->', left, right, TAB * 3)

    for left, right in itertools.product(teblocks, teblocks):
        sums += einsum_template('abji,abjicdlk,cdlk->', left, right, TAB * 3)


    print(f'''{PAD}return Polarizability.from_builder(
{PAD}{TAB}builder = lambda first, second: ({sums}
{PAD}{TAB})
{PAD})''')

if __name__ == "__main__":
    # imports()
    # calls()
    builder()
