import numpy as np
from progress.bar import Bar

def sarsa (env,num_samples,discount,step_size,epsilon,episode_maxlength=10000,init_Q=None,show_Qvalues=False) :

    def derive_greedy(actions,epsilon) :
        epsilon=epsilon*(1-((episode+1)/num_samples))
        greedy=np.zeros([len(actions)])
        for action in range(len(actions)) :
            if action==np.argmax(actions) :
                greedy[action]=1-epsilon
            else :
                greedy[action]=epsilon/(len(actions)-1)
        return greedy

    # Init Q values
    nS=env.observation_space.n
    nA=env.action_space.n
    if (init_Q==None).all() :
        Q=np.zeros([nS,nA])
    else :
        Q=init_Q
    # sample episodes
    with Bar('Sampling...',max = num_samples) as bar:
        for episode in range(num_samples) :
            if show_Qvalues : print(Q)
            state=env.reset()
            epsilon*(1-((episode+1)/num_samples))
            action=np.random.choice(nA,p=derive_greedy(Q[state],epsilon))
            done=False
            steps=0
            while not done and steps<episode_maxlength :
                new_state,reward,done,prob = env.step(action)
                new_action=np.random.choice(nA,p=derive_greedy(Q[state],epsilon))
                Q[state][action]+= step_size*(reward + discount*Q[new_state][new_action] - Q[state][action])
                state=new_state
                action=new_action
                steps+=1
            bar.next()

    # Derive pi as greedy wrt final Q
    pi = np.zeros([nS,nA])
    for ind,row in enumerate(Q) :
        for act in range(len(row)) :
            if act==np.argmax(row) :
                pi[ind][act]=1

    return pi

def q_learning (env,num_samples,discount,step_size,epsilon,episode_maxlength=10000,init_Q=None,show_Qvalues=False) :

    def derive_greedy(actions,epsilon) :
        epsilon=epsilon*(1-((episode+1)/num_samples))
        greedy=np.zeros([len(actions)])
        for action in range(len(actions)) :
            if action==np.argmax(actions) :
                greedy[action]=1-epsilon
            else :
                greedy[action]=epsilon/(len(actions)-1)
        return greedy

    # Init Q values
    nS=env.observation_space.n
    nA=env.action_space.n
    if (init_Q==None).all() :
        Q=np.zeros([nS,nA])
    else :
        Q=init_Q

    # sample episodes
    with Bar('Sampling...',max = num_samples) as bar:
        for episode in range(num_samples) :
            if show_Qvalues : print(Q)
            state=env.reset()
            done=False
            steps=0
            while not done and steps<episode_maxlength :
                action=np.random.choice(nA,p=derive_greedy(Q[state],epsilon))
                new_state,reward,done,prob = env.step(action)

                # Max(Q[new_state]) is used as estimator of the true q value for the best action
                # No guarantee that E(est)=q(A*)
                Q[state][action]+= step_size*(reward + discount*max(Q[new_state]) - Q[state][action])
                state=new_state
                steps+=1
            bar.next()

    # Derive pi as greedy wrt final Q
    pi = np.zeros([nS,nA])
    for s in range(nS) :
        pi[s][np.argmax(Q[s])]=1

    return pi

def double_q_learning (env,num_samples,discount,step_size,epsilon,episode_maxlength=10000,init_Q=None,show_Qvalues=False) :

    def derive_greedy(actions,epsilon) :
        epsilon=epsilon*(1-((episode+1)/num_samples))
        greedy=np.zeros([len(actions)])
        for action in range(len(actions)) :
            if action==np.argmax(actions) :
                greedy[action]=1-epsilon
            else :
                greedy[action]=epsilon/(len(actions)-1)
        return greedy

    # Init Q values
    nS=env.observation_space.n
    nA=env.action_space.n
    if (init_Q==None).all() :
        Q1=np.zeros([nS,nA])
        Q2=np.zeros([nS,nA])
    else :
        Q1=init_Q
        Q2=init_Q

    # sample episodes
    with Bar('Sampling...',max = num_samples) as bar:
        for episode in range(num_samples) :
            if show_Qvalues : print(Q1)
            state=env.reset()
            done=False
            steps=0
            while not done and steps<episode_maxlength :
                action=np.random.choice(nA,p=derive_greedy(Q1[state]+Q2[state],epsilon))
                new_state,reward,done,prob = env.step(action)

                if np.random.random() < 0.5 :
                    Q1[state][action]+= step_size*(reward + discount*Q2[new_state][np.argmax(Q1[new_state])] - Q1[state][action])
                else :
                    Q2[state][action]+= step_size*(reward + discount*Q1[new_state][np.argmax(Q2[new_state])] - Q2[state][action])
                state=new_state
                steps+=1
            bar.next()

    # Derive pi as greedy wrt final Q
    pi = np.zeros([nS,nA])
    for ind,row in enumerate(Q1+Q2) :
        for act in range(len(row)) :
            if act==np.argmax(row) :
                pi[ind][act]=1

    return pi

def expected_sarsa (env,num_samples,discount,step_size,epsilon,episode_maxlength=10000,init_Q=None,show_Qvalues=False) :

    def derive_greedy(actions,epsilon) :
        epsilon=epsilon*(1-((episode+1)/num_samples))
        greedy=np.zeros([len(actions)])
        for action in range(len(actions)) :
            if action==np.argmax(actions) :
                greedy[action]=1-epsilon
            else :
                greedy[action]=epsilon/(len(actions)-1)
        return greedy

    # Init Q values
    nS=env.observation_space.n
    nA=env.action_space.n
    if (init_Q==None).all() :
        Q=np.zeros([nS,nA])
    else :
        Q=init_Q

    # sample episodes
    with Bar('Sampling...',max = num_samples, redirect_stdout=True) as bar:
        for episode in range(num_samples) :
            if show_Qvalues : print(Q)
            state=env.reset()
            action=np.random.choice(nA,p=derive_greedy(Q[state],epsilon))
            done=False
            steps=0
            while not done and steps<episode_maxlength :
                new_state,reward,done,prob = env.step(action)
                new_action=np.random.choice(nA,p=derive_greedy(Q[state],epsilon))
                
                # Computing expected values for exp_sarsa rule
                exp=0
                greedy_new_state = derive_greedy(Q[new_state],epsilon)
                for a in range(nA) :
                    exp+=greedy_new_state[a]*Q[new_state][a]
                
                Q[state][action]+= step_size*(reward + discount*exp - Q[state][action])
                
                state=new_state
                action=new_action
                steps+=1
            bar.next()

    # Derive pi as greedy wrt final Q
    pi = np.zeros([nS,nA])
    for ind,row in enumerate(Q) :
        for act in range(len(row)) :
            if act==np.argmax(row) :
                pi[ind][act]=1

    return pi


def off_policy_mc_control (env,num_samples,gamma, epsilon,init_policy=None,episode_maxlength=1000000,low_reward_early_stopping=False) :
    S_n=env.observation_space.n
    A_n=env.action_space.n
    if init_policy == None :
        b=np.ones([S_n,A_n])/A_n
        Q=np.ones([S_n,A_n])/A_n
    else :
        b=np.loadtxt(init_policy, delimiter=',')
        Q=np.loadtxt(init_policy, delimiter=',')
    C=np.zeros([S_n,A_n])
    
    #Initializing greedy policy wrt initial Q
    pi=np.zeros([S_n,A_n])
    for ind,row in enumerate(Q) :
        pi[ind][np.argmax(row)]=1


    with Bar('Sampling',max=num_samples) as bar :
        for episode in range(num_samples) :
            
            # Sample an episode
            Samples=[]
            episode_length=0
            state=env.reset()
            done=False
            while not done and episode_maxlength>episode_length:
                action=np.random.choice(A_n,p=b[state])
                new_state, reward, done, prob = env.step(action)
                sample=[state, action, reward, new_state]
                Samples.append(sample)
                state=new_state
                episode_length+=1

                # stop the episode early when receiving terrible reward
                if low_reward_early_stopping :
                    if -100>=reward : done=True

            G=0
            W=1

            for sample in reversed(Samples) :
                G*=gamma
                G+=sample[2]
                C[sample[0]][sample[1]]+=W

                # Update Q
                Q[sample[0]][sample[1]]+=(W/C[sample[0]][sample[1]])*(G-Q[sample[0]][sample[1]])
                
                # # Update pi greedily wrt Q
                # for act in range(len(pi[sample[0]])) :
                #     if act==np.argmax(Q[sample[0]]) :
                #         pi[sample[0]][act]=1
                #     else : pi[sample[0]][act]=0

                # Update pi as epsilon-greedy policy
                for act in range(len(pi[sample[0]])) :
                    if act==np.argmax(Q[sample[0]]) :
                        pi[sample[0]][act]=1-epsilon*(1-((episode+1)/num_samples))
                    else : 
                        pi[sample[0]][act]=epsilon*(1-((episode+1)/num_samples))/(A_n-1)

                # Update b as epsilon-greedy policy
                for act in range(len(pi[sample[0]])) :
                    if act==np.argmax(Q[sample[0]]) :
                        b[sample[0]][act]=1-epsilon
                    else : 
                        b[sample[0]][act]=epsilon/(A_n-1)
                
                # if pi(s) != action taken in this step
                if np.argmax(pi[sample[0]]) != sample[1] :
                    continue

                W*=1/b[sample[0]][sample[1]]
            
            bar.next()
  
    return pi

def on_policy_mc_control (env,num_samples,gamma,epsilon,episode_maxlength=1000000,init_policy=None,low_reward_early_stopping=False) :
    S_n=env.observation_space.n
    A_n=env.action_space.n
    if init_policy == None :
        pi=np.ones([S_n,A_n])/A_n
    else :
        pi=np.loadtxt(init_policy, delimiter=',')
    Q=np.zeros([S_n,A_n])
    Returns=np.array(list(
        np.array(list([0,0.0] for i in range(A_n)))
        for j in range(S_n)))

    with Bar('Sampling...',max = num_samples) as bar:
        for episode in range(num_samples) :
            done=False
            state=env.reset()
            current_epsilon=epsilon*(1-((episode+1)/num_samples))

            # Sample an episode
            Samples=[]
            episode_length=0
            while not done and episode_maxlength>episode_length:
                action=np.random.choice(A_n,p=pi[state])
                new_state, reward, done, prob = env.step(action)
                sample=[state, action, reward, new_state]
                Samples.append(sample)
                state=new_state
                episode_length+=1
                if low_reward_early_stopping :
                    if -100>=reward : done=True

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
                        pi[index0][a]= 1-current_epsilon+(current_epsilon/A_n)
                    else :
                        pi[index0][a]= (current_epsilon/A_n)
            bar.next()
    return pi