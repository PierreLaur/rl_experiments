import numpy as np

def test_policy (pi,env,render=False) :
    A_n=env.action_space.n
    init_state=env.reset()
    if render : env.render()
    observation=env.step(np.random.choice(A_n,p=pi[init_state]))
    ret=0
    for i in range(100) :
        if render : env.render()
        observation=env.step(np.random.choice(A_n,p=pi[observation[0]]))
        ret+=observation[1]
        if observation[2]==True :
            return ret
    return ret

def evaluate_policy (pi,env, tries,render=False) :
    if type(pi) == type("") :
        print("Evaluating policy {}:".format(pi))
        pi = np.loadtxt(pi, delimiter=',')

    
    total=0.0
    for i in range(tries) :
        if render and i==tries-1 : total+=test_policy(pi,env,render=True)
        else : total+=test_policy(pi,env)
    total/=tries
    print("   Average return over {} tries : {}".format(tries,total))
    return total

def save_policy(policy,file_name) :
    np.savetxt(file_name, policy, delimiter=',')
    print("Saved policy to",file_name)
