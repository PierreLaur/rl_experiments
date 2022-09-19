from queue import PriorityQueue
import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt

def setup_plot(plot_color,plot_label,xlims,ylims) :
    axes = plt.gca()
    axes.set_xlim(*xlims)
    axes.set_ylim(*ylims)
    xdata = []
    ydata = []
    line, = axes.plot(xdata, ydata, plot_color,label=plot_label)
    axes.set_xlabel('Episodes')
    axes.set_ylabel('Average return')
    axes.set_title('Training TD algorithms')
    axes.legend()
    return axes, line, xdata, ydata

def plot(env,pi,line,xdata,ydata,episode,EWMA_factor=0) :
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    xdata.append(episode)

    ev=evaluate(env,pi,20,display=False)

    # optional parametrable moving average to smooth out the curve
    if episode != 0 :
        ydata.append(ydata[-1]*EWMA_factor + (1-EWMA_factor)*ev)
    else :
        ydata.append(ev)
    plt.draw()
    plt.pause(1e-17)
    return ev

def evaluate(env, policy, tries, render=False, display=True) :
        score=0
        for i in range(tries) :
            ret=0
            state=env.reset()
            done=False
            step=0
            while not done and step<100 :
                state,reward,done,_=env.step(np.random.choice(range(env.action_space.n),p=policy[state]))
                if render and i==tries-1 : env.render()
                ret+=reward
                step+=1
            score+=ret
        score/=tries
        if display : print("Average score over {} tries : {}".format(tries,score))
        return score

def derive_greedy(actions,epsilon,decay=False,epsilon_decay_rate=None,num_episodes=None,episode=None) :
        if decay : 
            epsilon=epsilon*(1-((episode+1)/num_episodes))
        greedy=np.zeros([len(actions)])
        for action in range(len(actions)) :
            if action==np.argmax(actions) :
                greedy[action]=1-epsilon
            else :
                greedy[action]=epsilon/(len(actions)-1)
        return greedy

def on_policy_mc_control (env,num_episodes,gamma,epsilon,episode_maxlength=10000,init_policy=None,low_reward_early_stopping=False,plot_return=False,plot_color='r',plot_label=None,xlims=(0,10000),ylims=(-100,100), EWMA_factor=0) :
    
    # To visualize return during training
    if plot_return :
        axes,line,xdata,ydata=setup_plot(plot_color,plot_label,xlims,ylims)
    
    nS=env.observation_space.n
    nA=env.action_space.n
    if init_policy == None :
        pi=np.ones([nS,nA])/nA
    else :
        pi=np.loadtxt(init_policy, delimiter=',')
    Q=np.zeros([nS,nA])
    Returns=np.array(list(
        np.array(list([0,0.0] for i in range(nA)))
        for j in range(nS)))

    with Bar('Sampling for On-Policy MC...',max = num_episodes) as bar:
        for episode in range(num_episodes) :
            done=False
            state=env.reset()
            current_epsilon=epsilon*(1-((episode+1)/num_episodes))

            # Sample an episode
            Samples=[]
            episode_length=0
            while not done and episode_maxlength>episode_length:
                action=np.random.choice(nA,p=pi[state])
                new_state, reward, done, prob = env.step(action)
                sample=[state, action, reward, new_state]
                Samples.append(sample)
                state=new_state
                episode_length+=1

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
                for a in range(nA) :
                    if a==A :
                        pi[index0][a]= 1-current_epsilon+(current_epsilon/nA)
                    else :
                        pi[index0][a]= (current_epsilon/nA)

            # to visualize the return while training
            if plot_return :
                if episode%100==0 :
                    pi = np.zeros([nS,nA])
                    for s in range(nS) :
                        pi[s][np.argmax(Q[s])]=1
                    plot(env,pi,line,xdata,ydata,episode,EWMA_factor)

            bar.next()
    return pi

def sarsa             (env,num_episodes,discount,step_size,epsilon,stepsize_decay_rate=0,exploration_decay_rate=0, episode_maxlength=10000,init_Q=None,show_Qvalues=False,plot_return=False,plot_color='r',plot_label=None,xlims=(0,10000),ylims=(-100,100), EWMA_factor=0) :

    # To visualize return during training
    if plot_return :
        axes,line,xdata,ydata=setup_plot(plot_color,plot_label,xlims,ylims)

    # Init Q values
    nS=env.observation_space.n
    nA=env.action_space.n
    if type(init_Q)==type(None) :
        Q=np.zeros([nS,nA])
    else :
        Q=init_Q.copy()
    # Interacting with the environment
    with Bar('Running SARSA...',max = num_episodes) as bar:
        for episode in range(num_episodes) :
            if show_Qvalues : print(Q)
            state=env.reset()
            epsilon*(1-((episode+1)/num_episodes))
            action=np.random.choice(nA,p=derive_greedy(Q[state],epsilon))
            done=False
            steps=0

            # exploration rate & step size decay
            current_epsilon=epsilon/(1+exploration_decay_rate*episode)
            current_step_size=step_size/(1+stepsize_decay_rate*episode)

            while not done and steps<episode_maxlength :
                new_state,reward,done,prob = env.step(action)
                new_action=np.random.choice(nA,p=derive_greedy(Q[state],epsilon))
                Q[state][action]+= step_size*(reward + discount*Q[new_state][new_action] - Q[state][action])
                state=new_state
                action=new_action
                steps+=1
            # to visualize the return while training
            if plot_return :
                if episode%100==0 :
                    pi = np.zeros([nS,nA])
                    for s in range(nS) :
                        pi[s][np.argmax(Q[s])]=1
                    plot(env,pi,line,xdata,ydata,episode,EWMA_factor)
            bar.next()

    # Derive pi as greedy wrt final Q
    pi = np.zeros([nS,nA])
    for ind,row in enumerate(Q) :
        for act in range(len(row)) :
            if act==np.argmax(row) :
                pi[ind][act]=1

    return pi

def q_learning        (env,num_episodes,discount,step_size,epsilon,stepsize_decay_rate=0,exploration_decay_rate=0, episode_maxlength=10000,init_Q=None,show_Qvalues=False,plot_return=False,plot_color='r',plot_label=None,xlims=(0,10000),ylims=(-100,100), EWMA_factor=0) :

    # To visualize return during training
    if plot_return :
        axes,line,xdata,ydata=setup_plot(plot_color,plot_label,xlims,ylims)

    # Init Q values
    nS=env.observation_space.n
    nA=env.action_space.n
    if type(init_Q)==type(None) :
        Q=np.zeros([nS,nA])
    else :
        Q=init_Q.copy()

    # Interacting with the environment
    with Bar('Running Q-learning...',max = num_episodes) as bar:
        for episode in range(num_episodes) :
            if show_Qvalues : print(Q)
            state=env.reset()
            done=False
            steps=0

            # exploration rate & step size decay
            current_epsilon=epsilon/(1+exploration_decay_rate*episode)
            current_step_size=step_size/(1+stepsize_decay_rate*episode)

            while not done and steps<episode_maxlength :
                
                
                action=np.random.choice(nA,p=derive_greedy(Q[state],current_epsilon))
                new_state,reward,done,prob = env.step(action)
                Q[state][action]+= current_step_size*(reward + discount*max(Q[new_state]) - Q[state][action])
                state=new_state
                steps+=1

            # to visualize the return while training
            if plot_return :
                if episode%100==0 :
                    pi = np.zeros([nS,nA])
                    for s in range(nS) :
                        pi[s][np.argmax(Q[s])]=1
                    ev=plot(env,pi,line,xdata,ydata,episode,EWMA_factor)

            bar.next()

    # Derive pi as greedy wrt final Q
    pi = np.zeros([nS,nA])
    for s in range(nS) :
        pi[s][np.argmax(Q[s])]=1

    return pi

def double_q_learning (env,num_episodes,discount,step_size,epsilon,stepsize_decay_rate=0,episode_maxlength=10000,init_Q=None,show_Qvalues=False,plot_return=False,plot_color='r',plot_label=None,xlims=(0,10000),ylims=(-100,100), EWMA_factor=0,optimal_return=None) :

    # To visualize return during training
    if plot_return :
        axes,line,xdata,ydata=setup_plot(plot_color,plot_label,xlims,ylims)

    # Init Q values
    nS=env.observation_space.n
    nA=env.action_space.n
    if type(init_Q)==type(None) :
        Q1=np.zeros([nS,nA])
        Q2=np.zeros([nS,nA])
    else :
        Q1=init_Q.copy()
        Q2=init_Q.copy()

    # Interacting with the environment
    with Bar('Running Double Q-Learning...',max = num_episodes) as bar:
        for episode in range(num_episodes) :
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

            # early exit procedure - evaluate the model every 20 episodes, exit when it get the optimal return
            if optimal_return!=None and episode%20==0 :
                policy=np.zeros([nS,nA])
                for s in range(nS) :
                    policy[s][np.argmax(Q1[s]+Q2[s])]=1
                if evaluate(env,policy,1,display=False)==optimal_return : 
                    print(" Achieved optimal policy in {} episodes.".format(episode))
                    break

            # to visualize the return while training
            if plot_return :
                if episode%100==0 :
                    pi = np.zeros([nS,nA])
                    for s in range(nS) :
                        pi[s][np.argmax(Q1[s]+Q2[s])]=1
                    plot(env,pi,line,xdata,ydata,episode,EWMA_factor)
            bar.next()

    # Derive pi as greedy wrt final Q
    pi = np.zeros([nS,nA])
    for ind,row in enumerate(Q1+Q2) :
        for act in range(len(row)) :
            if act==np.argmax(row) :
                pi[ind][act]=1

    return pi

def expected_sarsa    (env,num_episodes,discount,step_size,epsilon,stepsize_decay_rate=0,episode_maxlength=10000,init_Q=None,show_Qvalues=False,plot_return=False,plot_color='r',plot_label=None,xlims=(0,10000),ylims=(-100,100), EWMA_factor=0) :

    # To visualize return during training
    if plot_return :
        axes,line,xdata,ydata=setup_plot(plot_color,plot_label,xlims,ylims)

    # Init Q values
    nS=env.observation_space.n
    nA=env.action_space.n
    if type(init_Q)==type(None) :
        Q=np.zeros([nS,nA])
    else :
        Q=init_Q.copy()

    # Interacting with the environment
    with Bar('Running Expected SARSA...',max = num_episodes, redirect_stdout=True) as bar:
        for episode in range(num_episodes) :
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
            # to visualize the return while training
            if plot_return :
                if episode%100==0 :
                    pi = np.zeros([nS,nA])
                    for s in range(nS) :
                        pi[s][np.argmax(Q[s])]=1
                    plot(env,pi,line,xdata,ydata,episode,EWMA_factor)
            bar.next()

    # Derive pi as greedy wrt final Q
    pi = np.zeros([nS,nA])
    for ind,row in enumerate(Q) :
        for act in range(len(row)) :
            if act==np.argmax(row) :
                pi[ind][act]=1

    return pi

def dyna_q            (env,num_episodes,discount,step_size,epsilon,stepsize_decay_rate=0,episode_maxlength=10000,init_Q=None,show_Qvalues=False,plot_return=False,plot_color='r',plot_label=None,xlims=(0,10000),ylims=(-100,100), EWMA_factor=0,model_queries=5,optimal_return=None) :

    # To visualize return during training
    if plot_return :
        axes,line,xdata,ydata=setup_plot(plot_color,plot_label,xlims,ylims)

    nS=env.observation_space.n
    nA=env.action_space.n
    Q=np.zeros([nS,nA])
    Model=np.array(
        list([list([(0,0) for i in range(nA)]) for j in range(nS)])
        )
    visited=[]
    with Bar('Running Dyna-Q...',max=num_episodes) as bar :
        for episode in range(num_episodes) :
            state=env.reset()
            step=0
            done=False
            while not done and step<episode_maxlength :
                action=np.random.choice(range(nA),p=derive_greedy(Q[state],epsilon))
                new_state,reward,done,_=env.step(action)
                Q[state][action]+=step_size*(reward+discount*max(Q[new_state])-Q[state][action])
                visited.append((state,action))
                #dyna_q model update
                Model[state][action]=(reward,new_state)
                for i in range(model_queries) :
                    (sim_state,sim_action)=visited[np.random.choice(range(len(visited)))]
                    (sim_reward,sim_new_state)=Model[sim_state][sim_action]                
                    Q[sim_state][sim_action]+=step_size*(sim_reward+discount*max(Q[sim_new_state])-Q[sim_state][sim_action])
                state=new_state
                step+=1

            if optimal_return!=None and episode%20==0 :
                policy=np.zeros([nS,nA])
                for s in range(nS) :
                    policy[s][np.argmax(Q[s])]=1
                if evaluate(env,policy,1,display=False)==optimal_return : 
                    print(" Achieved optimal policy in {} episodes.".format(episode))
                    break
            # to visualize the return while training
            if plot_return :
                if episode%100==0 :
                    pi = np.zeros([nS,nA])
                    for s in range(nS) :
                        pi[s][np.argmax(Q[s])]=1
                    plot(env,pi,line,xdata,ydata,episode,EWMA_factor)
            bar.next()

    policy=np.zeros([nS,nA])
    for s in range(nS) :
        policy[s][np.argmax(Q[s])]=1

    return policy     


# not functional
def off_policy_mc_control (env,num_episodes,gamma,epsilon,init_policy=None,episode_maxlength=10000,low_reward_early_stopping=False) :
    nS=env.observation_space.n
    nA=env.action_space.n
    if init_policy == None :
        b=np.ones([nS,nA])/nA
        Q=np.ones([nS,nA])/nA
    else :
        b=np.loadtxt(init_policy, delimiter=',')
        Q=np.loadtxt(init_policy, delimiter=',')
    C=np.zeros([nS,nA])
    
    #Initializing greedy policy wrt initial Q
    pi=np.zeros([nS,nA])
    for ind,row in enumerate(Q) :
        pi[ind][np.argmax(row)]=1


    with Bar('Sampling for Off-Policy MC...',max=num_episodes) as bar :
        for episode in range(num_episodes) :
            
            # Sample an episode
            Samples=[]
            episode_length=0
            state=env.reset()
            done=False
            while not done and episode_maxlength>episode_length:
                action=np.random.choice(nA,p=b[state])
                new_state, reward, done, prob = env.step(action)
                sample=[state, action, reward, new_state]
                Samples.append(sample)
                state=new_state
                episode_length+=1

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
                        pi[sample[0]][act]=1-epsilon*(1-((episode+1)/num_episodes))
                    else : 
                        pi[sample[0]][act]=epsilon*(1-((episode+1)/num_episodes))/(nA-1)

                # Update b as epsilon-greedy policy
                for act in range(len(pi[sample[0]])) :
                    if act==np.argmax(Q[sample[0]]) :
                        b[sample[0]][act]=1-epsilon
                    else : 
                        b[sample[0]][act]=epsilon/(nA-1)
                
                # if pi(s) != action taken in this step
                if np.argmax(pi[sample[0]]) != sample[1] :
                    continue

                W*=1/b[sample[0]][sample[1]]
            
            bar.next()
  
    return pi

def prio_sweeping     (env,num_episodes,discount,step_size,epsilon,model_queries,episode_maxlength=10000, optimal_return=None, queue_threshold=0.5) :
    
    def derive_greedy(actions, epsilon) :
        nS=env.observation_space.n
        nA=env.action_space.n
        greedy=np.zeros([nA])
        for action,value in enumerate(actions) :
            if action==np.argmax(actions) :
                greedy[action]=1-epsilon
            else :
                greedy[action]=epsilon/(nA-1)
        return greedy

    def evaluate(env, policy, tries, render=False, display=True) :
        score=0
        for i in range(tries) :
            ret=0
            state=env.reset()
            done=False
            step=0
            while not done and step<100 :
                state,reward,done,_=env.step(np.random.choice(range(env.action_space.n),p=policy[state]))
                if render and i==tries-1 : env.render()
                ret+=reward
                step+=1
            score+=ret
        score/=tries
        if display : print("Average score over {} tries : {}".format(tries,score))
        return score

    nA=env.action_space.n
    nS=env.observation_space.n
    Q=np.zeros([nS,nA])
    Model={}
    Reversed_Model={s:set() for s in range(nS)}
    P=PriorityQueue()

    def update_queue(queue,state,action,reward,new_state,queue_threshold) :
        Q_update=abs(reward+discount*max(Q[new_state])-Q[state][action])
        if Q_update>queue_threshold :
            # put the negative Q update in the queue for priority reasons
            queue.put((-Q_update,(state,action)))
    
    with Bar(max=num_episodes) as bar :
        for episode in range(num_episodes) :
            state=env.reset()
            done=False
            step=0

            while not done and step<episode_maxlength :
                action=np.random.choice(range(nA),p=derive_greedy(Q[state],epsilon))
                new_state,reward,done,prob=env.step(action)
                #print("Visited",state,action,reward,new_state)
                Model[(state,action)]=(reward,new_state)
                Reversed_Model[new_state].add((state,action))
                update_queue(P,state,action,reward,new_state,queue_threshold)

                for i in range(model_queries) :
                    if P.qsize()==0 : break
                    (state,action)=P.get()[1]  
                    (reward,new_state)= Model[(state,action)]
                    #print("         Updated",state,action,reward,new_state)
                    Q[state][action] += step_size*(reward+discount*max(Q[new_state])-Q[state][action])

                    leading_states=Reversed_Model.get(state)
                    if len(leading_states)>0 :
                        for (state,action) in leading_states :
                            (reward,new_state)=Model[(state,action)]
                            update_queue(P,state,action,reward,new_state,queue_threshold)
                step+=1
                state=new_state

            # Early exit procedure
            if optimal_return!=None and episode%20==0 :
                policy=np.zeros([nS,nA])
                for s in range(nS) :
                    policy[s][np.argmax(Q[s])]=1
                eval_return=evaluate(env,policy,1,display=False)
                if eval_return==optimal_return : 
                    print(" Achieved optimal policy in {} episodes.".format(episode))
                    break
            
            bar.next()

    policy=np.zeros([nS,nA])
    for s in range(nS) :
        policy[np.argmax(Q[s])]=1

    return policy
