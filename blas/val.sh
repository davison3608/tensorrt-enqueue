valgrind --log-file=val.txt \
         --tool=memcheck   \
         --leak-check=yes      \
         --track-origins=yes   \
         --show-leak-kinds=all \
         --show-reachable=yes ./cumain

  

