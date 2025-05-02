+++
date = '2025-05-02T17:07:08+09:00'
draft = false
title = "index"
url = "/"
+++


# Hello world
안녕하세요, `키무야`입니다.



{{< plantuml id="eg" >}}
@startuml
actor me
queue "Task Queue" as queue
participant system


alt Weekday
    me -> queue : offer("출근")
    me -> queue : offer("일")
    me -> queue : offer("퇴근")
    me -> queue : offer("운동")
    me -> queue : offer("집으로 감")
else Weekend
    loop until awake
        me -> queue : offer("자는 중")
    end
    me -> queue : offer("운동")
    me -> queue : offer("집에 감")

end
loop while queue is not empty
    me -> queue : poll()
    queue -> system : execute Job
    system -> me : Job Complete
end
@enduml
{{< /plantuml >}}