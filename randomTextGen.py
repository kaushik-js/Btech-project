import random
class randText:
    def getText(self):
        #b = "about,above,add,after,again,air,all,almost,along,also,always,america,and,animal,another,answer,any,are,around,as,ask,at,away,back,be,because,before,began,begin,being,below,between,big,book,both,boy,but,call,came,can,car,carry,change,children,city,close,come,could,country,cut,day,did,different,do,does,don't,down,each,earth,end,enough,even,every,example,face,family,far,father,feet,few,find,first,follow,food,for,form,found,from,get,girl,give,go,good,got,great,group,grow,had,hand,hard,has,have,he,head,help,her,him,his,home,house,how,idea,if,important,in,indian,into,is,just,keep,kind,land,large,last,learn,left,life,light,like,line,list,little,long,look,made,make,man,many,may,mean,men,might,mile,miss,more,most,mother,mountain,move,much,must,my,name,near,need,never,new,next,night,not,now,number,often,oil,old,on,only,open,or,other,our,out,over,own,page,paper,part,people,picture,place,plant,play,point,put,question,quick,quickly,quite,read,really,river,run,said,same,saw,say,school,second,sentence,set,she,should,show,side,small,so,some,something,sometimes,song,soon,sound,spell,start,state,stop,story,study,such,take,talk,tell,than,that,the,then,these,they,thing,think,this,those,thought,through,time,together,took,tree,try,turn,under,until,up,us,use,very,walk,want,was,watch,water,way,we,well,went,what,when,which,while,white,who,why,will,with,without,word,work,world,would,year,you,young,your"
        b = 'mountain,america,basket,bucket,book,cluster,camera,dog,drop,direct,king,lemon,orange,black,monkey,next,pillow,robot,trip,office,support,search,elephant,children, mobile, customer,different,food, important, picture, people'
        rand_word = b.split(",")
        rand_word_len = len(rand_word)
        sz = 6
        rand_gen_text = []
        for i in range(sz):
            idx = random.randint(0, rand_word_len - 1)
            rand_gen_text.append(rand_word[idx])

        ret_str = ' '.join(rand_gen_text)
        return ret_str
