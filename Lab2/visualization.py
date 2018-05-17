import matplotlib.pyplot as plt
import numpy as np
import os


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class Visualize:
    def __init__(self, n_items):
        self.model_dict = {}
        self.model_dict2 = {}
        self.show(n_items)

    def get_dict(self,folder_name,filename,model_type):
        with open(os.path.join(folder_name,filename), 'r') as f:
            data = f.readlines()

        # overwrite if target word is repeated for all the dict
        word_dict = {}
        word_dict2 = {}
        for item in data:
            item = item.split()
            word_dict[item[1]] = item[3:]

            words = []
            w_score = []
            for i,word_i in enumerate(word_dict[item[1]]):
                if is_number(word_i):
                    w_score.append([' '.join(words), [float(word_i)]])
                    words = []
                else:
                    words.append(word_i)

            word_dict2[item[1]] = w_score

        word_dict3 = {}
        word_dict4 = {}
        for t_word in word_dict2:

            words = []
            score = []
            temp_dict = {}
            for x in word_dict2[t_word]:
                # print(x)
                words.append(x[0])
                score.append(x[1])
                temp_dict[x[0]] = x[1]
            word_dict3[t_word] = [words, score]
            word_dict4[t_word] = temp_dict

        self.model_dict[model_type] = word_dict3
        self.model_dict2[model_type] = word_dict4



    # i have list of numbers.
    # need to plot them in angles
    def plot(self,items,labels, model_type, target):
        N = len(items[0]) - 1
        angles = [(n/float(N) * 2 * np.pi) for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)

        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1],labels[0])
        ax.set_rlabel_position(0)

        plt.yticks([0.2,0.4,0.6, 0.8, 1],["0.2","0.4","0.6","0.8","1"], color="grey", size="7")
        plt.ylim(0,1)

        ax.plot(angles, items[0], '.', label=model_type[0], color='red')
        ax.plot(angles, items[1], '.', label=model_type[1], color='blue')
        ax.plot(angles, items[2], '.', label=model_type[2], color='green')
        ax.plot(angles, items[3], '.', label=model_type[3], color='orange')

        plt.legend(bbox_to_anchor=(0.92,0.9))
        plt.title("LST score of wrt "+str(target), )
        plt.show()

    def normalize(self,score):
        max_score = np.max(score)

        return [i/max_score for i in score]

    def show(self,n_items):
        # x= [3000,40000,24567,54324,10,5345,0]
        # y= [6,8,5,11,3,12,0]
        #
        # plot([x,y],y)

        folder_name = "model"
        filename1 = 'skip_add_lst.out'
        filename2 = 'skip_cosine_lst.out'
        filename3 = 'skip_mult_lst.out'
        filename4 = 'bsg_kl_lst.out'

        model_1 = 'skip_add'
        model_2 = 'skip_cosine'
        model_3 = 'skip_mult'
        model_4 = 'bsg'

        self.get_dict(folder_name, filename1, model_1)
        self.get_dict(folder_name, filename2, model_2)
        self.get_dict(folder_name, filename3, model_3)
        self.get_dict(folder_name, filename4,model_4)

        dict_1 = self.model_dict[model_1]
        dict_2 = self.model_dict[model_2]
        dict_3 = self.model_dict[model_3]
        dict_4 = self.model_dict[model_4]

        dict2_1 = self.model_dict2[model_1]
        dict2_2 = self.model_dict2[model_2]
        dict2_3 = self.model_dict2[model_3]
        dict2_4 = self.model_dict2[model_4]

        model_type = [model_1, model_2, model_3, model_4]

        counter = 0
        for key in dict_1:
            labels_1 = dict_1[key][0]
            score_1 = dict_1[key][1]
            # labels_2 = dict_2[key][0]
            score_2 = []
            # labels_3 = dict_3[key][0]
            score_3 = []
            # labels_4 = dict_4[key][0]
            score_4 = []

            for i,data in enumerate(labels_1):
                score_2.append(dict2_2[key][data])
                score_3.append(dict2_3[key][data])
                score_4.append(dict2_4[key][data])

            score_1 = self.normalize(score_1)
            score_2 = self.normalize(score_2)
            score_3 = self.normalize(score_3)
            score_4 = self.normalize(score_4)

            print(len(score_1), len(score_2))

            scores = [score_1, score_2, score_3, score_4]
            labels = [labels_1]

            self.plot(scores,labels, model_type, key)

            counter += 1
            if counter >= n_items:
                break




if __name__ == "__main__":

    Visualize(20)






