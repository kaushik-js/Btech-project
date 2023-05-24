class secure:
    def encode(self, pwd):
        x=[i for i in pwd] 
        x.reverse()
        str = ''
        for i in range(0,len(pwd)):
            str = str + '' + ord(x[i]).__str__() + '#%' + pwd[i] + '#%'
        return str
    def decode(self, pwd):
        x = pwd.split("#%")
        str = ''
        for i in range(1,len(x),2):
            str = str + '' + x[i]
        return str
