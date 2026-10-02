def say_hello():
    print("哈囉")

y = say_hello
print("這裡還沒印出哈囉,因為還沒按開關")
print(y)
y()   # 現在才是「按開關」,這行才會真的印出「哈囉」