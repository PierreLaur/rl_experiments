from random import seed
import gym
import numpy as np

env = gym.make ('FrozenLake-v1')
S_n=env.env.nS
A_n=env.env.nA
gamma=1
epsilon=0.05    
RANDOM_SEED=1
# env.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

def get_probs (V,arr) :
    res=[]
    for i in arr :
        res.append(i[0]*(i[2]+gamma*V[i[1]]))
    return sum(res)

def policy_evaluation (V, pi) :
    delta=1.0
    while delta>0.01  :
        delta=0.0
        for s in range(S_n) :
            oldV=V[s]
            
            # new V(s)
            V[s] = 0.0
            # for every action from s, every resulting state after a
            for a in range(A_n) :
                for sd in env.env.P[s][a] :
                    # 0:probability, 1:resulting state, 2:reward
                    V[s]+=pi[s][a]*sd[0]*(sd[2]+gamma*V[sd[1]])
            delta = max(delta,np.abs(oldV-V[s]))
    return V

def value_iteration (V, pi) :
    delta=1.0
    while delta>0.0001  :
        delta=0.0
        for s in range(S_n) :
            oldV=V[s]
            
            # candidates for V[s]
            cand=[0.0 for i in range(A_n)]
            # for every action from s, every resulting state after a
            for a in range(A_n) :
                for sd in env.env.P[s][a] :
                    # 0:probability, 1:resulting state, 2:reward
                    cand[a]+=(sd[0]*(sd[2]+gamma*V[sd[1]]))

            V[s]=max(cand)
            delta = max(delta,np.abs(oldV-V[s]))
    pi = policy_improvement(V,pi)
    return V, pi

def policy_improvement (V, pi) :
    for s in range(S_n) :
        oldAct = pi[s]
        sums = list(map(
                get_probs,[V for i in range(S_n)], list(env.env.P[s].values())
            ))
        for i in range(len(pi[s])) :
            if i==np.argmax(sums) :
                pi[s][i]=1.
            else : 
                pi[s][i]=0.
    return pi

def test_policy (pi) :
    init_state=env.reset()
    observation=env.step(np.random.choice(A_n,p=pi[init_state]))
    ret=0
    for i in range(100) :
        observation=env.step(np.random.choice(A_n,p=pi[observation[0]]))
        ret+=observation[1]
        if observation[2]==True :
            return ret
    return ret

def policy_iteration (V,pi) :
    stable=False
    while stable==False :
        stable=True
        oldPi=np.copy(pi)
        pi=policy_improvement(V,pi)
        if (oldPi != pi).any() :
            stable=False

        V=policy_evaluation(V,pi)
    return V,pi

def evaluate_policy (pi) :
    total=0.0
    for i in range(1000) :
        total+=test_policy(pi)
    return total

def on_policy_mc_control () :
    pi=np.ones([S_n,A_n])/A_n
    Q=np.zeros([S_n,A_n])
    Returns=np.array(list(
        np.array(list([0,0.0] for i in range(A_n)))
        for j in range(S_n)))
    
    for episode in range(10000) :
        done=False
        state=env.reset()

        # Sample an episode
        Samples=[]
        while not done :
            action=np.random.choice(A_n,p=pi[state])
            new_state, reward, done, prob = env.step(action)
            sample=[state, action, reward, new_state]
            Samples.append(sample)
            state=new_state

        G=0
        temp_returns={}
        for step in reversed(Samples) :
            G=gamma*G + step[2]
            temp_returns[(step[0],step[1])]=G

        for ret in temp_returns.items() :
            index0=ret[0][0]
            index1=ret[0][1]
            Returns[index0][index1][1]=ret[1]
            Returns[index0][index1][0]=Returns[index0][index1][0]+1
            Q[index0][index1]=Returns[index0][index1][1]/Returns[index0][index1][0]
            A=np.argmax(Q[index0],0)
            for a in range(A_n) :
                if a==A :
                    pi[index0][a]= 1-epsilon+(epsilon/A_n)
                else :
                    pi[index0][a]= (epsilon/A_n)
    return pi

# mc_pi=on_policy_mc_control()
# print(evaluate_policy(mc_pi))

pi=np.ones([S_n,A_n])/A_n
V=np.zeros([S_n])
print("Number of successful attempts (in 1000) for init policy :",evaluate_policy(pi))

V,pi=value_iteration(V,pi)
print("Number of successful attempts (in 1000) for value_iteration policy :",evaluate_policy(pi))